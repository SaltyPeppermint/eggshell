pub mod cost;
mod rules;

use std::sync::atomic::{AtomicBool, Ordering};

use egg::{Analysis, DidMerge, ENodeOrVar, Id, Language, PatternAst, Subst};
use log::warn;
use num::rational::Ratio;
use num::traits::Pow;
use num::{BigInt, BigRational, One, Signed, Zero};
use serde::{Deserialize, Serialize};

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

// Taken from https://github.com/herbie-fp/herbie/blob/main/egg-herbie/src/math.rs

egg::define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum Math {

        // constant-folding operators

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
        "neg" = Neg([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),
        "fabs" = Fabs([Id; 1]),
        "ceil" = Ceil([Id; 1]),
        "floor" = Floor([Id; 1]),
        "round" = Round([Id; 1]),
        "log" = Log([Id; 1]),
        "cbrt" = Cbrt([Id; 1]),

        Constant(BigRational),
        Symbol(egg::Symbol),
        Other(egg::Symbol, Vec<Id>),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConstantFold {
    pub unsound: AtomicBool,
    pub max_abs_exponent: Ratio<BigInt>,
    pub prune: bool,
}

impl Clone for ConstantFold {
    fn clone(&self) -> Self {
        let unsound = AtomicBool::new(self.unsound.load(Ordering::SeqCst));
        Self {
            unsound,
            max_abs_exponent: self.max_abs_exponent.clone(),
            prune: self.prune,
        }
    }
}

impl Default for ConstantFold {
    fn default() -> Self {
        Self {
            unsound: AtomicBool::new(false),
            // Avoid calculating extremely large numbers. 16 is somewhat arbitrary, even 0 passes
            // all tests.
            max_abs_exponent: Ratio::new(BigInt::from(16), BigInt::one()),
            prune: true,
        }
    }
}

impl Analysis<Math> for ConstantFold {
    type Data = Option<(BigRational, (PatternAst<Math>, Subst))>;

    fn make(egraph: &mut EGraph, enode: &Math, _: Id) -> Self::Data {
        let x = |id: &Id| egraph[*id].data.clone().map(|x| x.0);

        let r = match enode {
            Math::Constant(c) => c.clone(),

            // real
            Math::Add([a, b]) => x(a)? + x(b)?,
            Math::Sub([a, b]) => x(a)? - x(b)?,
            Math::Mul([a, b]) => x(a)? * x(b)?,
            // Math::Div([a, b]) => {
            //     if x(b)?.is_zero() {
            //         return None;
            //     }
            //     x(a)? / x(b)?
            // }
            Math::Div([a, b]) if !x(b)?.is_zero() => x(a)? / x(b)?,

            Math::Neg([a]) => -x(a)?,

            Math::Pow([a, b]) => {
                if is_zero(*a, egraph) {
                    // if x(b)?.is_positive() {
                    //     Ratio::zero()
                    // } else {
                    //     return None;
                    // }
                    (x(b)?.is_positive()).then(Ratio::zero)?
                } else if is_zero(*b, egraph) {
                    Ratio::one()
                } else if x(b)?.is_integer() && x(b)?.abs() <= egraph.analysis.max_abs_exponent {
                    Pow::pow(x(a)?, x(b)?.to_integer())
                } else {
                    return None;
                }
            }
            Math::Sqrt([a]) => {
                let inner_a = x(a)?;
                // if *inner_a.numer() > BigInt::zero() && *inner_a.denom() > BigInt::zero() {
                //     let s1 = inner_a.numer().sqrt();
                //     let s2 = inner_a.denom().sqrt();
                //     if &(&s1 * &s1) == inner_a.numer() && &(&s2 * &s2) == inner_a.denom() {
                //         Ratio::new(s1, s2)
                //     } else {
                //         return None;
                //     }
                // } else {
                //     return None;
                // }

                if *inner_a.numer() < BigInt::zero() || *inner_a.denom() < BigInt::zero() {
                    return None;
                }
                let s1 = inner_a.numer().sqrt();
                let s2 = inner_a.denom().sqrt();
                if &(&s1 * &s1) != inner_a.numer() || &(&s2 * &s2) != inner_a.denom() {
                    return None;
                }
                Ratio::new(s1, s2)
            }
            // Math::Log([a]) => {
            //     if x(a)? == Ratio::one() {
            //         Ratio::zero()
            //     } else {
            //         return None;
            //     }
            // }
            Math::Log([a]) => (x(a)? == Ratio::one()).then(Ratio::zero)?,
            // Math::Cbrt([a]) => {
            //     if x(a)? == Ratio::one() {
            //         Ratio::one()
            //     } else {
            //         return None;
            //     }
            // }
            Math::Cbrt([a]) => (x(a)? == Ratio::one()).then(Ratio::one)?,
            Math::Fabs([a]) => x(a)?.abs(),
            Math::Floor([a]) => x(a)?.floor(),
            Math::Ceil([a]) => x(a)?.ceil(),
            Math::Round([a]) => x(a)?.round(),

            _ => return None,
        };

        let mut pattern = PatternAst::<Math>::default();
        let mut var_counter = 0;
        let mut subst = Subst::default();
        enode.for_each(|child| {
            if let Some(constant) = x(&child) {
                pattern.add(ENodeOrVar::ENode(Math::Constant(constant)));
            } else {
                let var = format!("?{var_counter}").parse().unwrap();
                pattern.add(ENodeOrVar::Var(var));
                subst.insert(var, child);
                var_counter += 1;
            }
        });
        let mut counter = 0;
        let mut head = enode.clone();
        head.update_children(|_child| {
            let res = Id::from(counter);
            counter += 1;
            res
        });
        pattern.add(ENodeOrVar::ENode(head));
        Some((r, (pattern, subst)))
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        match (&to, from) {
            (None, None) => DidMerge(false, false),
            (Some(_), None) => DidMerge(false, true), // no update needed
            (None, Some(c)) => {
                *to = Some(c);
                DidMerge(true, false)
            }
            (Some(a), Some(ref b)) => {
                if a.0 != b.0 && !self.unsound.swap(true, Ordering::SeqCst) {
                    warn!("Bad merge detected: {} != {}", a.0, b.0);
                }
                DidMerge(false, false)
            }
        }
    }

    fn modify(egraph: &mut EGraph, class_id: Id) {
        let class = &mut egraph[class_id];
        if let Some((c, (pat, subst))) = class.data.clone() {
            egraph.union_instantiations(
                &pat,
                &format!("{c}").parse().unwrap(),
                &subst,
                "metadata-eval".to_owned(),
            );
        }
    }
}

fn is_zero(id: Id, egraph: &EGraph) -> bool {
    egraph[id]
        .data
        .as_ref()
        .is_some_and(|data| data.0.is_zero())
}

#[derive(Debug, Copy, Clone)]
pub enum HerbieRules {
    Ruleset121,
    Ruleset242,
    Ruleset704,
}

#[must_use]
pub fn rules(ruleset: HerbieRules) -> Vec<Rewrite> {
    match ruleset {
        HerbieRules::Ruleset121 => rules::rules_121(),
        HerbieRules::Ruleset242 => rules::rules_242(),
        HerbieRules::Ruleset704 => rules::rules_704(),
    }
}
