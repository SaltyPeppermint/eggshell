mod data;
mod rules;

use egg::{define_language, Analysis, DidMerge, Id, Subst, Symbol, Var};
use serde::Serialize;

use super::{Trs, TrsError};
use data::HalideData;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<HalideMath, HalideConstFold>;
type Rewrite = egg::Rewrite<HalideMath, HalideConstFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize)]
    pub enum HalideMath {
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
        Bool(bool),
        Constant(i64),
        Symbol(Symbol),
    }
}

/// Enabling Constant Folding through the Analysis of egg.
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct HalideConstFold;

impl Analysis<HalideMath> for HalideConstFold {
    type Data = Option<HalideData>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        match (to.as_mut(), from) {
            (None, Some(_)) => {
                *to = from;
                DidMerge(true, false)
            }
            (Some(_), None) => DidMerge(false, true),
            (Some(_), Some(_)) | (None, None) => DidMerge(false, false),
            // (Some(a), Some(b)) => match (a, b) {
            //     (HalideData::Int(i_a), HalideData::Int(i_b)) => {
            //         let cmp = (*i_a).cmp(&i_b);
            //         match cmp {
            //             Ordering::Less => DidMerge(false, true),
            //             Ordering::Equal => DidMerge(false, false),
            //             Ordering::Greater => {
            //                 *to = from;
            //                 DidMerge(true, false)
            //             }
            //         }
            //     }
            //     (HalideData::Bool(b_a), HalideData::Bool(b_b)) => {
            //         if *b_a == b_b {
            //             DidMerge(false, false)
            //         } else {
            //             panic!("Tried to merge false with true!")
            //         }
            //     }
            //     _ => panic!("Tried to merge truth value with constant!"),
            // },
        }
    }

    fn make(egraph: &EGraph, enode: &HalideMath) -> Self::Data {
        let xi = |i: &Id| egraph[*i].data.map(|d| i64::try_from(d).unwrap());
        let xb = |i: &Id| egraph[*i].data.map(|d| bool::try_from(d).unwrap());
        // let tv = |i: &Id| egraph[*i].data.map(|d: HalideData| d.as_bool());
        Some(match enode {
            HalideMath::Constant(c) => HalideData::Int(*c),
            HalideMath::Add([a, b]) => (xi(a)? + xi(b)?).into(),
            HalideMath::Sub([a, b]) => (xi(a)? - xi(b)?).into(),
            HalideMath::Mul([a, b]) => (xi(a)? * xi(b)?).into(),
            HalideMath::Div([a, b]) => {
                // Important to check otherwise integer division loss or div 0 error
                if xi(b)? != 0 && xi(a)? % xi(b)? == 0 {
                    (xi(a)? / xi(b)?).into()
                } else {
                    return None;
                }
            }
            HalideMath::Max([a, b]) => std::cmp::max(xi(a)?, xi(b)?).into(),
            HalideMath::Min([a, b]) => std::cmp::min(xi(a)?, xi(b)?).into(),
            HalideMath::Mod([a, b]) => {
                if xi(b)? == 0 {
                    HalideData::Int(0)
                } else {
                    (xi(a)? % xi(b)?).into()
                }
            }

            HalideMath::Lt([a, b]) => (xi(a)? < xi(b)?).into(),
            HalideMath::Gt([a, b]) => (xi(a)? > xi(b)?).into(),
            HalideMath::Let([a, b]) => (xi(a)? <= xi(b)?).into(),
            HalideMath::Get([a, b]) => (xi(a)? >= xi(b)?).into(),
            HalideMath::Eq([a, b]) => (xi(a)? == xi(b)?).into(),
            HalideMath::IEq([a, b]) => (xi(a)? != xi(b)?).into(),

            HalideMath::Not(a) => (!xb(a)?).into(),
            HalideMath::And([a, b]) => (xb(a)? && xb(b)?).into(),
            HalideMath::Or([a, b]) => (xb(a)? || xb(b)?).into(),

            _ => return None,
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = match c {
                HalideData::Int(i) => egraph.add(HalideMath::Constant(i)),
                HalideData::Bool(b) => egraph.add(HalideMath::Bool(b)),
            };

            // let _ =
            egraph.union_trusted(id, added, format!("eclass {id} contained analysis {c:?}"));
            let _ = egraph.union(id, added);

            egraph[id].nodes.retain(egg::Language::is_leaf);

            // assert!(
            //     !egraph[id].nodes.is_empty(),
            //     "empty eclass! {:#?}",
            //     egraph[id]
            // );
            // if !check_leaves(&egraph[id]) {
            //     println!(" ");
            //     let rec_exprs = &egraph[id]
            //         .leaves()
            //         .map(|v| RecExpr::from(vec![v.to_owned()]))
            //         .collect::<Vec<_>>();

            //     let mut expl = egraph.explain_equivalence(&rec_exprs[0], &rec_exprs[1]);
            //     expl.check_proof(&Halide::rules(&Ruleset::BugRules));
            //     println!("PROOF CHECKS OUT");

            //     println!("{}", &expl.get_flat_string());
            // }

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

// fn check_leaves<L, D>(class: &EClass<L, D>) -> bool
// where
//     L: egg::Language,
// {
//     let mut leaves = class.leaves();
//     if let Some(first) = leaves.next() {
//         if leaves.all(|l| l == first) {
//             return true;
//         }
//     }
//     false
// }

/// Checks if a constant is positive
#[allow(clippy::missing_panics_doc)]
pub fn is_const_pos(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get the constant
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
        // Check if any of the representations of ths constant (nodes inside its eclass) is positive
        // egraph[subst[var]].data.iter().any(|n| match n {
        //     HalideMath::Constant(c) => c > &0,
        //     _ => false,
        // })
        // NEW CORRECT

        match egraph[subst[var]].data {
            Some(HalideData::Int(x)) => x > 0,
            _ => false,
        }
    }
}

/// Checks if a constant is negative
#[allow(clippy::missing_panics_doc)]
pub fn is_const_neg(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // Check if any of the representations of ths constant (nodes inside its eclass) is negative
        // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
        // egraph[subst[var]].nodes.iter().any(|n| match n {
        //     HalideMath::Constant(c) => c < &0,
        //     _ => false,
        // })
        // NEW CORRECT

        match egraph[subst[var]].data {
            Some(HalideData::Int(x)) => x < 0,
            _ => false,
        }
    }
}

/// Checks if a constant is equals zero
#[allow(clippy::missing_panics_doc)]
pub fn is_not_zero(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
    // let zero = HalideMath::Constant(0);
    // // Check if any of the representations of the constant (nodes inside its eclass) is zero
    // move |egraph, _, subst| !egraph[subst[var]].nodes.contains(&zero)
    // NEW CORRECT
    move |egraph, _, subst| match egraph[subst[var]].data {
        Some(HalideData::Int(x)) => x != 0,
        _ => false,
    }
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

        let data_1 = egraph[subst[var_1]].data;
        let data_2 = egraph[subst[var_2]].data;

        match (data_1, data_2) {
            (Some(HalideData::Int(c_1)), Some(HalideData::Int(c_2))) => match comp {
                "<" => c_1 < c_2,
                "<a" => c_1 < c_2.abs(),
                "<=" => c_1 <= c_2,
                "<=+1" => c_1 <= (c_2 + 1),
                "<=a" => c_1 <= c_2.abs(),
                "<=-a" => c_1 <= (-c_2.abs()),
                "<=-a+1" => c_1 <= (1 - c_2.abs()),
                ">" => c_1 > c_2,
                ">a" => c_1 > c_2.abs(),
                ">=" => c_1 >= c_2,
                ">=a" => c_1 >= (c_2.abs()),
                ">=a-1" => c_1 >= (c_2.abs() - 1),
                "!=" => c_1 != c_2,
                "%0" => (c_2 != 0) && (c_1 % c_2 == 0),
                "!%0" => (c_2 != 0) && (c_1 % c_2 != 0),
                "%0<" => (c_2 > 0) && (c_1 % c_2 == 0),
                "%0>" => (c_2 < 0) && (c_1 % c_2 == 0),
                _ => false,
            },
            _ => false,
        }
    }
}

/// Enum for the Ruleset to use
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize)]
pub enum Ruleset {
    Arithmetic,
    BugRules,
    Full,
}

impl TryFrom<String> for Ruleset {
    type Error = TrsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            "arithmetic" | "Arithmetic" | "ARITHMETIC" => Ok(Self::Arithmetic),
            _ => Err(TrsError::BadRulesetName(value)),
        }
    }
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Halide;

impl Trs for Halide {
    type Language = HalideMath;
    type Analysis = HalideConstFold;
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
            Ruleset::BugRules => [
                (&*add_rules),
                // (&*div_rules),
                // (&*modulo_rules),
                (&*mul_rules),
                (&*sub_rules),
                // (&*max_rules),
                // (&*min_rules),
                (&*lt_rules),
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
}

#[cfg(test)]
mod tests {
    use crate::eqsat::{Eqsat, EqsatConfBuilder};
    use crate::utils::AstSize2;

    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_expr = vec!["( == 0 0 )".parse().unwrap()];
        let rules = Halide::rules(&Ruleset::Full);

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!(HalideMath::Bool(true), term[0.into()]);
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_expr = vec!["( == 1 0 )".parse().unwrap()];
        let rules = Halide::rules(&Ruleset::Full);

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!(HalideMath::Bool(false), term[0.into()]);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn halide_false_expr() {
        let expr = vec![
            "( < ( + ( + ( * v0 35 ) v1 ) 35 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
                .parse()
                .unwrap(),
        ];
        let rules = Halide::rules(&Ruleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn halide_false_expr2() {
        let expr = vec!["( < ( + ( * v0 35 ) v1 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
            .parse()
            .unwrap()];
        let rules = Halide::rules(&Ruleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn halide_false_expr3() {
        let expr = vec!["( < ( * v0 35 ) ( * ( + v0 1 ) 35 ) )".parse().unwrap()];
        let rules = Halide::rules(&Ruleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }
}
