mod rules;

use egg::{define_language, Analysis, DidMerge, Id, Subst, Symbol, Var};
use serde::Serialize;

use super::Trs;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<MathEquations, EquationConstFold>;
type Rewrite = egg::Rewrite<MathEquations, EquationConstFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize)]
    pub enum MathEquations {
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "max" = Max([Id; 2]),
        "min" = Min([Id; 2]),
        "<" = Lt([Id; 2]),
        ">" = Gt([Id; 2]),
        "!" = Not(Id),
        "<=" = Let([Id;2]),
        ">=" = Get([Id;2]),
        "==" = Eq([Id; 2]),
        "!=" = IEq([Id; 2]),
        "||" = Or([Id; 2]),
        "&&" = And([Id; 2]),
        Constant(i64),
        Symbol(Symbol),
    }
}

/// Enabling Constant Folding through the Analysis of egg.
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct EquationConstFold;

impl Analysis<MathEquations> for EquationConstFold {
    type Data = Option<i64>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, egg::merge_min)
    }

    fn make(egraph: &EGraph, enode: &MathEquations) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref();
        Some(match enode {
            MathEquations::Constant(c) => *c,
            MathEquations::Add([a, b]) => x(a)? + x(b)?,
            MathEquations::Sub([a, b]) => x(a)? - x(b)?,
            MathEquations::Mul([a, b]) => {
                let x_a = x(a)?;
                let x_b = x(b)?;
                println!("Found a multiplication: {x_a} * {x_b}");
                x(a)? * x(b)?
            }
            MathEquations::Div([a, b]) if *x(b)? != 0 => x(a)? / x(b)?,
            MathEquations::Max([a, b]) => std::cmp::max(*x(a)?, *x(b)?),
            MathEquations::Min([a, b]) => std::cmp::min(*x(a)?, *x(b)?),
            MathEquations::Not(a) => i64::from(*x(a)? == 0),
            MathEquations::Lt([a, b]) => i64::from(x(a)? < x(b)?),
            MathEquations::Gt([a, b]) => i64::from(x(a)? > x(b)?),
            MathEquations::Let([a, b]) => i64::from(x(a)? <= x(b)?),
            MathEquations::Get([a, b]) => i64::from(x(a)? >= x(b)?),
            MathEquations::Mod([a, b]) => {
                if *x(b)? == 0 {
                    0
                } else {
                    x(a)? % x(b)?
                }
            }
            MathEquations::Eq([a, b]) => i64::from(x(a)? == x(b)?),
            MathEquations::IEq([a, b]) => i64::from(x(a)? != x(b)?),
            MathEquations::And([a, b]) => i64::from(!(*x(a)? == 0 || *x(b)? == 0)),
            MathEquations::Or([a, b]) => i64::from(*x(a)? == 1 || *x(b)? == 1),

            _ => return None,
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data;
        if let Some(c) = data {
            // if egraph.are_explanations_enabled() {
            //     egraph.union_instantiations(
            //         &pat,
            //         &format!("{}", c).parse().unwrap(),
            //         &Default::default(),
            //         "constant_fold".to_string(),
            //     );
            // } else {
            let added = egraph.add(MathEquations::Constant(c));
            let _ = egraph.union(id, added);
            // }
            // to not prune, comment this out
            egraph[id].nodes.retain(egg::Language::is_leaf);

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

/// Checks if a constant is positive
pub(crate) fn is_const_pos(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get the constant
    let var = var.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // Check if any of the representations of ths constant (nodes inside its eclass) is positive
        egraph[subst[var]].nodes.iter().any(|n| match n {
            MathEquations::Constant(c) => c > &0,
            _ => false,
        })
    }
}

/// Checks if a constant is negative
fn is_const_neg(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        //Check if any of the representations of ths constant (nodes inside its eclass) is negative
        egraph[subst[var]].nodes.iter().any(|n| match n {
            MathEquations::Constant(c) => c < &0,
            _ => false,
        })
    }
}

/// Checks if a constant is equals zero
fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    let zero = MathEquations::Constant(0);
    // Check if any of the representations of the constant (nodes inside its eclass) is zero
    move |egraph, _, subst| !egraph[subst[var]].nodes.contains(&zero)
}

/// Compares two constants c0 and c1
fn compare_constants(
    // first constant
    var: &str,
    // 2nd constant
    var1: &str,
    // the comparison we're checking
    comp: &'static str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get constants
    let var: Var = var.parse().unwrap();
    let var1: Var = var1.parse().unwrap();

    move |egraph, _, subst| {
        // Get the eclass of the first constant then match the values of its enodes to check if one of them proves the coming conditions
        egraph[subst[var1]].nodes.iter().any(|n1| match n1 {
            // Get the eclass of the second constant then match it to c1
            MathEquations::Constant(c1) => egraph[subst[var]].nodes.iter().any(|n| match n {
                // match the comparison then do it
                MathEquations::Constant(c) => match comp {
                    "<" => c < c1,
                    "<a" => c < &c1.abs(),
                    "<=" => c <= c1,
                    "<=+1" => c <= &(c1 + 1),
                    "<=a" => c <= &c1.abs(),
                    "<=-a" => c <= &(-c1.abs()),
                    "<=-a+1" => c <= &(1 - c1.abs()),
                    ">" => c > c1,
                    ">a" => c > &c1.abs(),
                    ">=" => c >= c1,
                    ">=a" => c >= &(c1.abs()),
                    ">=a-1" => c >= &(c1.abs() - 1),
                    "!=" => c != c1,
                    "%0" => (*c1 != 0) && (c % c1 == 0),
                    "!%0" => (*c1 != 0) && (c % c1 != 0),
                    "%0<" => (*c1 > 0) && (c % c1 == 0),
                    "%0>" => (*c1 < 0) && (c % c1 == 0),
                    _ => false,
                },
                _ => false,
            }),
            _ => false,
        })
    }
}

/// Enum for the Ruleset to use
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Ruleset {
    Arithmetic,
    Full,
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Halide;

impl Trs for Halide {
    type Language = MathEquations;
    type Analysis = EquationConstFold;
    type Rulesets = Ruleset;

    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[allow(clippy::similar_names)]
    #[must_use]
    fn rules(ruleset_class: &Ruleset) -> Vec<Rewrite> {
        let add_rules = self::rules::add::add();
        let and_rules = self::rules::and::and();
        let andor_rules = self::rules::andor::andor();
        let div_rules = self::rules::div::div();
        let eq_rules = self::rules::eq::eq();
        let ineq_rules = self::rules::ineq::ineq();
        let lt_rules = self::rules::lt::lt();
        let max_rules = self::rules::max::max();
        let min_rules = self::rules::min::min();
        let modulo_rules = self::rules::modulo::modulo();
        let mul_rules = self::rules::mul::mul();
        let not_rules = self::rules::not::not();
        let or_rules = self::rules::or::or();
        let sub_rules = self::rules::sub::sub();

        match ruleset_class {
            // Class that only contains arithmetic operations' rules
            Ruleset::Arithmetic => [
                &add_rules[..],
                &div_rules[..],
                &modulo_rules[..],
                &mul_rules[..],
                &sub_rules[..],
            ]
            .concat(),
            // All the rules
            Ruleset::Full => [
                &add_rules[..],
                &and_rules[..],
                &andor_rules[..],
                &div_rules[..],
                &eq_rules[..],
                &ineq_rules[..],
                &lt_rules[..],
                &max_rules[..],
                &min_rules[..],
                &modulo_rules[..],
                &mul_rules[..],
                &not_rules[..],
                &or_rules[..],
                &sub_rules[..],
            ]
            .concat(),
        }
    }

    #[must_use]
    fn maximum_ruleset() -> Self::Rulesets {
        Ruleset::Full
    }

    // #[must_use]
    // fn prove_goals() -> Vec<egg::Pattern<Self::Language>> {
    //     let goals = ["1".parse().unwrap(), "0".parse().unwrap()];
    //     goals.to_vec()
    // }
}
