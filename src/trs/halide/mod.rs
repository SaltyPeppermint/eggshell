mod data;
mod rules;

use std::{cmp::Ordering, fmt::Display};

use egg::{define_language, Analysis, DidMerge, Id, Symbol};
use serde::Serialize;

use super::{SymbolIter, Trs, TrsError};
use crate::typing::{Type, Typeable, TypingInfo};
use data::HalideData;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<HalideExpr, ConstantFold>;
type Rewrite = egg::Rewrite<HalideExpr, ConstantFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize)]
    pub enum HalideExpr {
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

impl SymbolIter for HalideExpr {
    fn raw_symbols() -> &'static [(&'static str, usize)] {
        &[
            ("+", 2),
            ("-", 2),
            ("*", 2),
            ("/", 2),
            ("%", 2),
            ("max", 2),
            ("min", 2),
            ("<", 2),
            (">", 2),
            ("!", 1),
            ("<=", 2),
            (">=", 2),
            ("==", 2),
            ("!=", 2),
            ("||", 2),
            ("&&", 2),
            ("true", 0),
            ("false", 0),
        ]
    }
}

impl Typeable for HalideExpr {
    type Type = HalideType;

    fn type_info(&self) -> TypingInfo<Self::Type> {
        match self {
            // Primitive types
            Self::Bool(_) => TypingInfo::new(Self::Type::Boolean, Self::Type::Top),
            Self::Constant(_) => TypingInfo::new(Self::Type::Integer, Self::Type::Top),
            Self::Symbol(_) => TypingInfo::new(Self::Type::Top, Self::Type::Top),

            // Fns of type int
            Self::Add(_)
            | Self::Sub(_)
            | Self::Mul(_)
            | Self::Div(_)
            | Self::Mod(_)
            | Self::Max(_)
            | Self::Min(_) => TypingInfo::new(Self::Type::Integer, Self::Type::Integer),

            // Fns of type bool
            Self::Lt(_) | Self::Gt(_) | Self::Let(_) | Self::Get(_) => {
                TypingInfo::new(Self::Type::Boolean, Self::Type::Integer)
            }
            Self::Or(_) | Self::And(_) | Self::Not(_) => {
                TypingInfo::new(Self::Type::Boolean, Self::Type::Boolean)
            }

            // Fns of generic type
            Self::Eq(_) | HalideExpr::IEq(_) => {
                TypingInfo::new(Self::Type::Boolean, Self::Type::Top)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum HalideType {
    Integer,
    Boolean,
    Top,
    Bottom,
}

impl Display for HalideType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer => write!(f, "Integer"),
            Self::Boolean => write!(f, "Boolean"),
            Self::Top => write!(f, "Top"),
            Self::Bottom => write!(f, "Bottom"),
        }
    }
}

impl PartialOrd for HalideType {
    #[expect(clippy::match_same_arms)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Can't compare int and bool
            (Self::Integer, Self::Boolean) | (Self::Boolean, Self::Integer) => None,
            // Compare to self
            (Self::Boolean, Self::Boolean)
            | (Self::Integer, Self::Integer)
            | (Self::Top, Self::Top)
            | (Self::Bottom, Self::Bottom) => Some(Ordering::Equal),
            // Top is greater than bool and int
            (Self::Boolean | Self::Integer, Self::Top) => Some(Ordering::Less),
            (Self::Top, Self::Integer | Self::Boolean) => Some(Ordering::Greater),

            // Bottom Type is smaller than everything else
            (Self::Integer | Self::Boolean | Self::Top, Self::Bottom) => Some(Ordering::Greater),
            (Self::Bottom, Self::Integer | Self::Boolean | Self::Top) => Some(Ordering::Less),
        }
    }
}

impl Type for HalideType {
    fn top() -> Self {
        Self::Top
    }

    fn bottom() -> Self {
        Self::Bottom
    }
}

/// Enabling Constant Folding through the Analysis of egg.
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<HalideExpr> for ConstantFold {
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

    fn make(egraph: &EGraph, enode: &HalideExpr) -> Self::Data {
        let xi = |i: &Id| egraph[*i].data.map(|d| i64::try_from(d).unwrap());
        let xb = |i: &Id| egraph[*i].data.map(|d| bool::try_from(d).unwrap());
        // let tv = |i: &Id| egraph[*i].data.map(|d: HalideData| d.as_bool());
        Some(match enode {
            HalideExpr::Constant(c) => HalideData::Int(*c),
            HalideExpr::Add([a, b]) => (xi(a)? + xi(b)?).into(),
            HalideExpr::Sub([a, b]) => (xi(a)? - xi(b)?).into(),
            HalideExpr::Mul([a, b]) => (xi(a)? * xi(b)?).into(),
            HalideExpr::Div([a, b]) => {
                // Important to check otherwise integer division loss or div 0 error
                if xi(b)? != 0 && xi(a)? % xi(b)? == 0 {
                    (xi(a)? / xi(b)?).into()
                } else {
                    return None;
                }
            }
            HalideExpr::Max([a, b]) => std::cmp::max(xi(a)?, xi(b)?).into(),
            HalideExpr::Min([a, b]) => std::cmp::min(xi(a)?, xi(b)?).into(),
            HalideExpr::Mod([a, b]) => {
                if xi(b)? == 0 {
                    HalideData::Int(0)
                } else {
                    (xi(a)? % xi(b)?).into()
                }
            }

            HalideExpr::Lt([a, b]) => (xi(a)? < xi(b)?).into(),
            HalideExpr::Gt([a, b]) => (xi(a)? > xi(b)?).into(),
            HalideExpr::Let([a, b]) => (xi(a)? <= xi(b)?).into(),
            HalideExpr::Get([a, b]) => (xi(a)? >= xi(b)?).into(),
            HalideExpr::Eq([a, b]) => (xi(a)? == xi(b)?).into(),
            HalideExpr::IEq([a, b]) => (xi(a)? != xi(b)?).into(),

            HalideExpr::Not(a) => (!xb(a)?).into(),
            HalideExpr::And([a, b]) => (xb(a)? && xb(b)?).into(),
            HalideExpr::Or([a, b]) => (xb(a)? || xb(b)?).into(),

            _ => return None,
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = match c {
                HalideData::Int(i) => egraph.add(HalideExpr::Constant(i)),
                HalideData::Bool(b) => egraph.add(HalideExpr::Bool(b)),
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
enum HalideRuleset {
    Arithmetic,
    BugRules,
    Full,
}

impl HalideRuleset {
    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[expect(clippy::similar_names)]
    #[must_use]
    fn rules(self) -> Vec<Rewrite> {
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

        match self {
            // Class that only contains arithmetic operations' rules
            HalideRuleset::Arithmetic => [
                (&*add_rules),
                (&*div_rules),
                (&*modulo_rules),
                (&*mul_rules),
                (&*sub_rules),
            ]
            .concat(),
            HalideRuleset::BugRules => [
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
            HalideRuleset::Full => [
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

impl TryFrom<String> for HalideRuleset {
    type Error = TrsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            "arithmetic" | "Arithmetic" | "ARITHMETIC" => Ok(Self::Arithmetic),
            _ => Err(Self::Error::BadRulesetName(value)),
        }
    }
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Halide;

impl Trs for Halide {
    type Language = HalideExpr;
    type Analysis = ConstantFold;

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        HalideRuleset::Full.rules()
    }
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr};

    use crate::eqsat::{Eqsat, EqsatConfBuilder};
    use crate::typing::typecheck_expr;

    use super::*;

    #[test]
    fn eqsat_solved_true() {
        let false_expr = vec!["( == 0 0 )".parse().unwrap()];
        let rules = HalideRuleset::rules(HalideRuleset::Full);

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideExpr::Bool(true), term[0.into()]);
    }

    #[test]
    fn eqsat_solved_false() {
        let false_expr = vec!["( == 1 0 )".parse().unwrap()];
        let rules = HalideRuleset::rules(HalideRuleset::Full);

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideExpr::Bool(false), term[0.into()]);
    }

    #[test]
    fn expl_1() {
        let expr = vec![
            "( < ( + ( + ( * v0 35 ) v1 ) 35 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
                .parse()
                .unwrap(),
        ];
        let rules = HalideRuleset::rules(HalideRuleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    fn expl_2() {
        let expr = vec!["( < ( + ( * v0 35 ) v1 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
            .parse()
            .unwrap()];
        let rules = HalideRuleset::rules(HalideRuleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn expl_3() {
        let expr = vec!["( < ( * v0 35 ) ( * ( + v0 1 ) 35 ) )".parse().unwrap()];
        let rules = HalideRuleset::rules(HalideRuleset::BugRules);

        let eqsat =
            Eqsat::<Halide>::new(expr).with_conf(EqsatConfBuilder::new().explanation(true).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_1() {
        let expr: RecExpr<HalideExpr> = "( < ( * v0 35 ) false )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_err());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_2() {
        let expr: RecExpr<HalideExpr> = "( max ( == v0 v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_err());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn correct_typing() {
        let expr: RecExpr<HalideExpr> = "( && ( == ( * v0 35 ) v1 ) true )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_ok());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn bool_typing() {
        let expr: RecExpr<HalideExpr> = "( && ( == ( * v0 35 ) v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Boolean, tc);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn int_typing() {
        let expr: RecExpr<HalideExpr> = "( - ( + ( * v0 35 ) v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Integer, tc);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_inferred() {
        let expr: RecExpr<HalideExpr> = "( max ( + v0 v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Integer, tc);
    }
}
