mod rules;

use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

use egg::{
    Analysis, DidMerge, ENodeOrVar, Id, Language, Pattern, PatternAst, Subst, define_language,
};
use num::rational::Ratio;
use num::{BigInt, BigRational, One, Signed, Zero};
use num_traits::Pow;
use serde::{Deserialize, Serialize};

use super::RewriteSystem;

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

// Taken from https://github.com/herbie-fp/herbie/blob/main/egg-herbie/src/math.rs

define_language! {
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

    fn make(egraph: &mut EGraph, enode: &Math) -> Self::Data {
        let x = |id: &Id| egraph[*id].data.clone().map(|x| x.0);
        let is_zero = |id: &Id| match egraph[*id].data.as_ref() {
            Some(data) => data.0.is_zero(),
            None => false,
        };

        Some((
            match enode {
                Math::Constant(c) => c.clone(),

                // real
                Math::Add([a, b]) => x(a)? + x(b)?,
                Math::Sub([a, b]) => x(a)? - x(b)?,
                Math::Mul([a, b]) => x(a)? * x(b)?,
                Math::Div([a, b]) => {
                    if x(b)?.is_zero() {
                        return None;
                    }
                    x(a)? / x(b)?
                }
                Math::Neg([a]) => -x(a)?,
                Math::Pow([a, b]) if is_zero(a) => {
                    if x(b)?.is_positive() {
                        Ratio::new(BigInt::zero(), BigInt::one())
                    } else {
                        return None;
                    }
                }
                Math::Pow([a, b]) if is_zero(b) => Ratio::new(BigInt::one(), BigInt::one()),
                Math::Pow([a, b])
                    if x(b)?.is_integer() && x(b)?.abs() <= egraph.analysis.max_abs_exponent =>
                {
                    Pow::pow(x(a)?, x(b)?.to_integer())
                }
                // Falls through to None default case
                // Math::Pow([a, b]) => {
                //     return None;
                // }
                Math::Sqrt([a]) => {
                    let inner_a = x(a)?;
                    if *inner_a.numer() > BigInt::zero() && *inner_a.denom() > BigInt::zero() {
                        let s1 = inner_a.numer().sqrt();
                        let s2 = inner_a.denom().sqrt();
                        let is_perfect =
                            &(&s1 * &s1) == inner_a.numer() && &(&s2 * &s2) == inner_a.denom();
                        if is_perfect {
                            Ratio::new(s1, s2)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
                Math::Log([a]) => {
                    if x(a)? == Ratio::new(BigInt::one(), BigInt::one()) {
                        Ratio::new(BigInt::zero(), BigInt::one())
                    } else {
                        return None;
                    }
                }
                Math::Cbrt([a]) => {
                    if x(a)? == Ratio::new(BigInt::one(), BigInt::one()) {
                        Ratio::new(BigInt::one(), BigInt::one())
                    } else {
                        return None;
                    }
                }
                Math::Fabs([a]) => x(a)?.abs(),
                Math::Floor([a]) => x(a)?.floor(),
                Math::Ceil([a]) => x(a)?.ceil(),
                Math::Round([a]) => x(a)?.round(),

                _ => return None,
            },
            {
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
                (pattern, subst)
            },
        ))
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
                    log::warn!("Bad merge detected: {} != {}", a.0, b.0);
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

pub fn mk_rules(tuples: &[(&str, &str, &str)]) -> Vec<Rewrite> {
    tuples
        .iter()
        .map(|(name, left, right)| {
            let left = Pattern::from_str(left).unwrap();
            let right = Pattern::from_str(right).unwrap();
            Rewrite::new(*name, left, right).unwrap()
        })
        .collect()
}

// #[derive(Default, Debug, Clone, Copy, Serialize)]
// pub struct Arithmetic;

// impl RewriteSystem for Arithmetic {
//     type Language = Math;
//     type Analysis = ConstantFold;

//     fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
//         self::rules::rules()
//     }
// }
