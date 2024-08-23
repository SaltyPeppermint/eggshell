mod rules;

use egg::{define_language, Analysis, DidMerge, Id, Subst, Symbol, Var};
use serde::Serialize;

use super::Trs;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<MathEquation, EquationConstFold>;
type Rewrite = egg::Rewrite<MathEquation, EquationConstFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize)]
    pub enum MathEquation {
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

impl Analysis<MathEquation> for EquationConstFold {
    type Data = Option<i64>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, egg::merge_max)
    }

    fn make(egraph: &EGraph, enode: &MathEquation) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref();
        Some(match enode {
            MathEquation::Constant(c) => *c,
            MathEquation::Add([a, b]) => x(a)? + x(b)?,
            MathEquation::Sub([a, b]) => x(a)? - x(b)?,
            MathEquation::Mul([a, b]) => {
                let x_a = x(a)?;
                let x_b = x(b)?;
                println!("Found a multiplication: {x_a} * {x_b}");
                x(a)? * x(b)?
            }
            MathEquation::Div([a, b]) if *x(b)? != 0 => x(a)? / x(b)?,
            MathEquation::Max([a, b]) => std::cmp::max(*x(a)?, *x(b)?),
            MathEquation::Min([a, b]) => std::cmp::min(*x(a)?, *x(b)?),
            MathEquation::Not(a) => i64::from(*x(a)? == 0),
            MathEquation::Lt([a, b]) => i64::from(x(a)? < x(b)?),
            MathEquation::Gt([a, b]) => i64::from(x(a)? > x(b)?),
            MathEquation::Let([a, b]) => i64::from(x(a)? <= x(b)?),
            MathEquation::Get([a, b]) => i64::from(x(a)? >= x(b)?),
            MathEquation::Mod([a, b]) => {
                if *x(b)? == 0 {
                    0
                } else {
                    x(a)? % x(b)?
                }
            }
            MathEquation::Eq([a, b]) => i64::from(x(a)? == x(b)?),
            MathEquation::IEq([a, b]) => i64::from(x(a)? != x(b)?),
            MathEquation::And([a, b]) => i64::from(!(*x(a)? == 0 || *x(b)? == 0)),
            MathEquation::Or([a, b]) => i64::from(*x(a)? == 1 || *x(b)? == 1),

            _ => return None,
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = egraph.add(MathEquation::Constant(c));
            let _ = egraph.union(id, added);
            egraph[id].nodes.retain(egg::Language::is_leaf);
            dbg!(egraph[id].leaves().collect::<Vec<_>>());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

/// Checks if a constant is positive
#[allow(clippy::missing_panics_doc)]
pub fn is_const_pos(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get the constant
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // Check if any of the representations of ths constant (nodes inside its eclass) is positive
        egraph[subst[var]].nodes.iter().any(|n| match n {
            MathEquation::Constant(c) => c > &0,
            _ => false,
        })
    }
}

/// Checks if a constant is negative
#[allow(clippy::missing_panics_doc)]
pub fn is_const_neg(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // Check if any of the representations of ths constant (nodes inside its eclass) is negative
        egraph[subst[var]].nodes.iter().any(|n| match n {
            MathEquation::Constant(c) => c < &0,
            _ => false,
        })
    }
}

/// Checks if a constant is equals zero
#[allow(clippy::missing_panics_doc)]
pub fn is_not_zero(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    let zero = MathEquation::Constant(0);
    // Check if any of the representations of the constant (nodes inside its eclass) is zero
    move |egraph, _, subst| !egraph[subst[var]].nodes.contains(&zero)
}

/// Compares two constants c0 and c1
#[allow(clippy::missing_panics_doc)]
pub fn compare_constants(
    // first constant
    var_str_1: &str,
    // 2nd constant
    var_str_2: &str,
    // the comparison we're checking
    comp: &'static str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get constants
    let var_1: Var = var_str_1.parse().unwrap();
    let var_2: Var = var_str_2.parse().unwrap();

    move |egraph, _, subst| {
        // Get the eclass of the first constant then match the values of its enodes to check if one of them proves the coming conditions
        egraph[subst[var_2]].nodes.iter().any(|n1| match n1 {
            // Get the eclass of the second constant then match it to c1
            MathEquation::Constant(c1) => egraph[subst[var_1]].nodes.iter().any(|n| match n {
                // match the comparison then do it
                MathEquation::Constant(c) => match comp {
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
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Ruleset {
    Arithmetic,
    Full,
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Halide;

impl Trs for Halide {
    type Language = MathEquation;
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
                (&*add_rules),
                (&*div_rules),
                (&*modulo_rules),
                (&*mul_rules),
                (&*sub_rules),
            ]
            .concat(),
            // All the rules
            Ruleset::Full => [
                (&*add_rules),
                (&*and_rules),
                (&*andor_rules),
                (&*div_rules),
                (&*eq_rules),
                (&*ineq_rules),
                (&*lt_rules),
                (&*max_rules),
                (&*min_rules),
                (&*modulo_rules),
                (&*mul_rules),
                (&*not_rules),
                (&*or_rules),
                (&*sub_rules),
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
