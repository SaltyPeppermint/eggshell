mod data;
mod rules;

use std::{cmp::Ordering, fmt::Display};

use egg::{define_language, Analysis, DidMerge, Id, RecExpr, Symbol};
use serde::Serialize;

use super::{Trs, TrsError, Typeable};
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

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

/// Enum for the Ruleset to use
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

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum HalideType {
    Integer,
    Boolean,
    Top,
}

impl Display for HalideType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HalideType::Integer => write!(f, "Integer"),
            HalideType::Boolean => write!(f, "Boolean"),
            HalideType::Top => write!(f, "Top"),
        }
    }
}

impl PartialOrd for HalideType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Can't compare int and bool
            (HalideType::Integer, HalideType::Boolean)
            | (HalideType::Boolean, HalideType::Integer) => None,
            // Compare to self
            (HalideType::Boolean, HalideType::Boolean)
            | (HalideType::Integer, HalideType::Integer)
            | (HalideType::Top, HalideType::Top) => Some(Ordering::Equal),
            // Top is greater than bool and int
            (HalideType::Boolean | HalideType::Integer, HalideType::Top) => Some(Ordering::Less),
            (HalideType::Top, HalideType::Integer | HalideType::Boolean) => Some(Ordering::Greater),
        }
    }
}

impl Typeable for HalideMath {
    type Type = HalideType;

    fn type_node(&self, expr: &RecExpr<HalideMath>) -> Result<Self::Type, TrsError> {
        match self {
            // Primitive types
            HalideMath::Bool(_) => Ok(HalideType::Boolean),
            HalideMath::Constant(_) => Ok(HalideType::Integer),
            HalideMath::Symbol(_) => Ok(HalideType::Top),

            // Fns of type int
            HalideMath::Add(children)
            | HalideMath::Sub(children)
            | HalideMath::Mul(children)
            | HalideMath::Div(children)
            | HalideMath::Mod(children)
            | HalideMath::Max(children)
            | HalideMath::Min(children) => {
                Typeable::infer_type(HalideType::Integer, children, expr)
            }

            // Fns of type bool
            HalideMath::Lt(children)
            | HalideMath::Gt(children)
            | HalideMath::Let(children)
            | HalideMath::Get(children)
            | HalideMath::Or(children)
            | HalideMath::And(children) => {
                Typeable::infer_type(HalideType::Boolean, children, expr)
            }
            HalideMath::Not(child) => Typeable::infer_type(HalideType::Boolean, &[*child], expr),

            // Fns of generic type
            HalideMath::Eq(children) | HalideMath::IEq(children) => {
                Typeable::infer_type(HalideType::Top, children, expr)
            }
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
    #[expect(clippy::similar_names)]
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
