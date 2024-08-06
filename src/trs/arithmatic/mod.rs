mod rules;

use egg::{define_language, Analysis, DidMerge, Id, Language, PatternAst, Subst, Symbol};
use ordered_float::NotNan;
use serde::Serialize;

use super::Trs;

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

pub type Constant = NotNan<f64>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize)]
    pub enum Math {
        "d" = Diff([Id; 2]),
        "i" = Integral([Id; 2]),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
        "ln" = Ln(Id),
        "sqrt" = Sqrt(Id),

        "sin" = Sin(Id),
        "cos" = Cos(Id),

        Constant(Constant),
        Symbol(Symbol),
    }
}

pub struct MathCostFn;
impl egg::CostFunction<Math> for MathCostFn {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Diff(..) | Math::Integral(..) => 100,
            _ => 1,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<Math> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Math>)>;

    fn make(egraph: &EGraph, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            Math::Constant(c) => (*c, format!("{c}").parse().unwrap()),
            Math::Add([a, b]) => (
                x(a)? + x(b)?,
                format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Sub([a, b]) => (
                x(a)? - x(b)?,
                format!("(- {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Mul([a, b]) => (
                x(a)? * x(b)?,
                format!("(* {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Div([a, b]) if x(b) != Some(NotNan::new(0.0).unwrap()) => (
                x(a)? / x(b)?,
                format!("(/ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{c}").parse().unwrap(),
                    &Subst::default(),
                    "constant_fold".to_owned(),
                );
            } else {
                let added = egraph.add(Math::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn is_const_or_distinct_var(
    var_str1: &str,
    var_str2: &str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var1 = var_str1.parse().unwrap();
    let var2 = var_str2.parse().unwrap();
    move |egraph, _, subst| {
        egraph.find(subst[var1]) != egraph.find(subst[var2])
            && (egraph[subst[var1]].data.is_some()
                || egraph[subst[var1]]
                    .nodes
                    .iter()
                    .any(|n| matches!(n, Math::Symbol(..))))
    }
}

fn is_const(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.is_some()
}

fn is_sym(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]]
            .nodes
            .iter()
            .any(|n| matches!(n, Math::Symbol(..)))
    }
}

fn is_not_zero(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) != 0.0
        } else {
            true
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Ruleset {
    Full,
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Arithmatic;

impl Trs for Arithmatic {
    type Language = Math;
    type Analysis = ConstantFold;
    type Rulesets = Ruleset;

    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[allow(clippy::similar_names)]
    #[must_use]
    fn rules(_ruleset_class: &Ruleset) -> Vec<Rewrite> {
        self::rules::rules()
    }

    #[must_use]
    fn maximum_ruleset() -> Self::Rulesets {
        Ruleset::Full
    }

    // #[must_use]
    // fn prove_goals() -> Vec<egg::Pattern<Self::Language>> {
    //     panic!("THERE ARE NO PROVE GOALS HERE")
    // }
}
