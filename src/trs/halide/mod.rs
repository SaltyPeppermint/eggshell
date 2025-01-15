mod data;
mod rules;

use std::cmp::Ordering;
use std::fmt::Display;

use egg::{define_language, Analysis, DidMerge, Id, Symbol};
use serde::{Deserialize, Serialize};

use super::{LanguageManager, MetaInfo, SymbolType, TermRewriteSystem, TrsError};
use crate::typing::{Type, Typeable, TypingInfo};
use data::HalideData;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<HalideLang, ConstantFold>;
type Rewrite = egg::Rewrite<HalideLang, ConstantFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize, Deserialize)]
    pub enum HalideLang {
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
        Number(i64),
        Symbol(Symbol),
    }
}

impl MetaInfo for HalideLang {
    fn manager(variable_names: Vec<String>) -> LanguageManager<Self> {
        LanguageManager::new(
            vec![
                HalideLang::Add([0.into(), 0.into()]),
                HalideLang::Sub([0.into(), 0.into()]),
                HalideLang::Mul([0.into(), 0.into()]),
                HalideLang::Div([0.into(), 0.into()]),
                HalideLang::Mod([0.into(), 0.into()]),
                HalideLang::Max([0.into(), 0.into()]),
                HalideLang::Min([0.into(), 0.into()]),
                HalideLang::Lt([0.into(), 0.into()]),
                HalideLang::Gt([0.into(), 0.into()]),
                HalideLang::Not(0.into()),
                HalideLang::Let([0.into(), 0.into()]),
                HalideLang::Get([0.into(), 0.into()]),
                HalideLang::Eq([0.into(), 0.into()]),
                HalideLang::IEq([0.into(), 0.into()]),
                HalideLang::Or([0.into(), 0.into()]),
                HalideLang::And([0.into(), 0.into()]),
                HalideLang::Bool(true),
                HalideLang::Number(0),
                // We do not include symbols!
                // HalideLang::Symbol(Symbol::new("")),
            ],
            variable_names,
        )
    }

    #[expect(clippy::cast_precision_loss)]
    fn symbol_type(&self) -> SymbolType {
        match self {
            HalideLang::Bool(value) => SymbolType::Constant(if *value { 1.0 } else { 0.0 }),
            HalideLang::Number(value) => SymbolType::Constant(*value as f64),
            HalideLang::Symbol(name) => SymbolType::Variable(name.as_str()),
            _ => SymbolType::Operator,
        }
    }

    fn into_symbol(name: String) -> Self {
        HalideLang::Symbol(name.into())
    }
}

impl Typeable for HalideLang {
    type Type = HalideType;

    fn type_info(&self) -> TypingInfo<Self::Type> {
        match self {
            // Primitive types
            Self::Bool(_) => TypingInfo::new(Self::Type::Boolean, Self::Type::Top),
            Self::Number(_) => TypingInfo::new(Self::Type::Integer, Self::Type::Top),
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
            Self::Eq(_) | HalideLang::IEq(_) => {
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

impl Analysis<HalideLang> for ConstantFold {
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

    fn make(egraph: &mut EGraph, enode: &HalideLang) -> Self::Data {
        let xi = |i: &Id| egraph[*i].data.map(|d| i64::try_from(d).unwrap());
        let xb = |i: &Id| egraph[*i].data.map(|d| bool::try_from(d).unwrap());
        // let tv = |i: &Id| egraph[*i].data.map(|d: HalideData| d.as_bool());
        Some(match enode {
            HalideLang::Number(c) => HalideData::Int(*c),
            HalideLang::Add([a, b]) => (xi(a)? + xi(b)?).into(),
            HalideLang::Sub([a, b]) => (xi(a)? - xi(b)?).into(),
            HalideLang::Mul([a, b]) => (xi(a)? * xi(b)?).into(),
            HalideLang::Div([a, b]) => {
                // Important to check otherwise integer division loss or div 0 error
                if xi(b)? != 0 && xi(a)? % xi(b)? == 0 {
                    (xi(a)? / xi(b)?).into()
                } else {
                    return None;
                }
            }
            HalideLang::Max([a, b]) => std::cmp::max(xi(a)?, xi(b)?).into(),
            HalideLang::Min([a, b]) => std::cmp::min(xi(a)?, xi(b)?).into(),
            HalideLang::Mod([a, b]) => {
                if xi(b)? == 0 {
                    HalideData::Int(0)
                } else {
                    (xi(a)? % xi(b)?).into()
                }
            }

            HalideLang::Lt([a, b]) => (xi(a)? < xi(b)?).into(),
            HalideLang::Gt([a, b]) => (xi(a)? > xi(b)?).into(),
            HalideLang::Let([a, b]) => (xi(a)? <= xi(b)?).into(),
            HalideLang::Get([a, b]) => (xi(a)? >= xi(b)?).into(),
            HalideLang::Eq([a, b]) => (xi(a)? == xi(b)?).into(),
            HalideLang::IEq([a, b]) => (xi(a)? != xi(b)?).into(),

            HalideLang::Not(a) => (!xb(a)?).into(),
            HalideLang::And([a, b]) => (xb(a)? && xb(b)?).into(),
            HalideLang::Or([a, b]) => (xb(a)? || xb(b)?).into(),

            _ => return None,
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = match c {
                HalideData::Int(i) => egraph.add(HalideLang::Number(i)),
                HalideData::Bool(b) => egraph.add(HalideLang::Bool(b)),
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
pub enum HalideRuleset {
    Arithmetic,
    BugRules,
    Full,
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

impl Halide {
    #[expect(clippy::similar_names)]
    #[must_use]
    pub fn rules(ruleset: HalideRuleset) -> Vec<Rewrite> {
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

        match ruleset {
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

impl TermRewriteSystem for Halide {
    type Language = HalideLang;
    type Analysis = ConstantFold;

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        Self::rules(HalideRuleset::Full)
    }
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr};

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::typing::typecheck_expr;

    use super::*;

    #[test]
    fn eqsat_solved_true() {
        let false_expr = vec!["( == 0 0 )".parse().unwrap()];
        let rules = Halide::rules(HalideRuleset::Full);

        let result = Eqsat::new(StartMaterial::RecExprs(false_expr)).run(&rules);
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideLang::Bool(true), expr[0.into()]);
    }

    #[test]
    fn eqsat_solved_false() {
        let false_expr = vec!["( == 1 0 )".parse().unwrap()];
        let rules = Halide::rules(HalideRuleset::Full);

        let result = Eqsat::new(StartMaterial::RecExprs(false_expr)).run(&rules);
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideLang::Bool(false), expr[0.into()]);
    }

    #[test]
    fn expl_1() {
        let expr = vec![
            "( < ( + ( + ( * v0 35 ) v1 ) 35 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
                .parse()
                .unwrap(),
        ];
        let rules = Halide::rules(HalideRuleset::BugRules);

        let eqsat = Eqsat::new(StartMaterial::RecExprs(expr))
            .with_conf(EqsatConf::builder().explanation(true).iter_limit(3).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    fn expl_2() {
        let expr = vec!["( < ( + ( * v0 35 ) v1 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
            .parse()
            .unwrap()];
        let rules = Halide::rules(HalideRuleset::BugRules);

        let eqsat = Eqsat::new(StartMaterial::RecExprs(expr))
            .with_conf(EqsatConf::builder().explanation(true).iter_limit(3).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn expl_3() {
        let expr = vec!["( < ( * v0 35 ) ( * ( + v0 1 ) 35 ) )".parse().unwrap()];
        let rules = Halide::rules(HalideRuleset::BugRules);

        let eqsat = Eqsat::new(StartMaterial::RecExprs(expr))
            .with_conf(EqsatConf::builder().explanation(true).iter_limit(3).build());
        let _ = eqsat.run(&rules);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_1() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) false )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_err());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_2() {
        let expr: RecExpr<HalideLang> = "( max ( == v0 v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_err());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn correct_typing() {
        let expr: RecExpr<HalideLang> = "( && ( == ( * v0 35 ) v1 ) true )".parse().unwrap();
        let tc = typecheck_expr(&expr);
        assert!(tc.is_ok());
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn bool_typing() {
        let expr: RecExpr<HalideLang> = "( && ( == ( * v0 35 ) v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Boolean, tc);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn int_typing() {
        let expr: RecExpr<HalideLang> = "( - ( + ( * v0 35 ) v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Integer, tc);
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn false_typing_inferred() {
        let expr: RecExpr<HalideLang> = "( max ( + v0 v1 ) v2 )".parse().unwrap();
        let tc = typecheck_expr(&expr).unwrap();
        assert_eq!(HalideType::Integer, tc);
    }
}
