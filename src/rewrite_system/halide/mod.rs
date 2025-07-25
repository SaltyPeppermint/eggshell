mod data;
mod rules;

use egg::{Analysis, DidMerge, Id, Symbol, define_language};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{LangExtras, RewriteSystem, RewriteSystemError, SymbolInfo, SymbolType};
use data::HalideData;

// Defining aliases to reduce code.
type EGraph = egg::EGraph<HalideLang, ConstantFold>;
type Rewrite = egg::Rewrite<HalideLang, ConstantFold>;

// Definition of the language used.
define_language! {
    #[derive(Serialize, Deserialize, EnumDiscriminants, EnumCount)]
    #[strum_discriminants(derive(EnumIter))]
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

impl LangExtras for HalideLang {
    fn symbol_info(&self) -> SymbolInfo {
        let id = HalideLangDiscriminants::iter()
            .position(|x| x == self.into())
            .unwrap();

        match self {
            HalideLang::Symbol(name) => SymbolInfo::new(id, SymbolType::Variable(name.to_string())),
            HalideLang::Bool(value) => SymbolInfo::new(id, SymbolType::Constant(value.to_string())),
            HalideLang::Number(value) => {
                SymbolInfo::new(id, SymbolType::Constant(value.to_string()))
            }
            _ => SymbolInfo::new(id, SymbolType::Operator),
        }
    }

    fn operators() -> Vec<&'static str> {
        vec![
            "+", "-", "*", "/", "%", "max", "min", "<", ">", "!", "<=", ">=", "==", "!=", "||",
            "&&",
        ]
    }

    const MAX_ARITY: usize = 2;
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
    type Error = RewriteSystemError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            "arithmetic" | "Arithmetic" | "ARITHMETIC" => Ok(Self::Arithmetic),
            _ => Err(Self::Error::BadRulesetName(value)),
        }
    }
}

/// Halide Rewrite System implementation
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

impl RewriteSystem for Halide {
    type Language = HalideLang;
    type Analysis = ConstantFold;

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        Self::rules(HalideRuleset::Full)
    }
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr};

    use super::*;
    use crate::eqsat::{Eqsat, EqsatConf};

    #[test]
    fn eqsat_solved_true() {
        let true_expr: RecExpr<HalideLang> = "( == 0 0 )".parse().unwrap();
        let rules = Halide::rules(HalideRuleset::Full);

        let result = Eqsat::new((&true_expr).into(), &rules).run();
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideLang::Bool(true), expr[0.into()]);
    }

    #[test]
    fn eqsat_solved_false() {
        let false_expr: RecExpr<HalideLang> = "( == 1 0 )".parse().unwrap();
        let rules = Halide::rules(HalideRuleset::Full);

        let result = Eqsat::new((&false_expr).into(), &rules).run();
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(HalideLang::Bool(false), expr[0.into()]);
    }

    #[test]
    fn expl_1() {
        let expr: RecExpr<HalideLang> =
            "( < ( + ( + ( * v0 35 ) v1 ) 35 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
                .parse()
                .unwrap();
        let rules = Halide::rules(HalideRuleset::BugRules);
        let conf = EqsatConf::builder().explanation(true).iter_limit(3).build();

        let eqsat = Eqsat::new((&expr).into(), &rules).with_conf(conf);
        let _ = eqsat.run();
    }

    #[test]
    fn expl_2() {
        let expr: RecExpr<HalideLang> = "( < ( + ( * v0 35 ) v1 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
            .parse()
            .unwrap();
        let rules = Halide::rules(HalideRuleset::BugRules);
        let conf = EqsatConf::builder().explanation(true).iter_limit(3).build();

        let eqsat = Eqsat::new((&expr).into(), &rules).with_conf(conf);
        let _ = eqsat.run();
    }

    #[test]
    // #[should_panic(expected = "Different leaves in eclass 1: {Constant(0), Constant(35)}")]
    fn expl_3() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 1 ) 35 ) )".parse().unwrap();
        let rules = Halide::rules(HalideRuleset::BugRules);
        let conf = EqsatConf::builder().explanation(true).iter_limit(3).build();

        let eqsat = Eqsat::new((&expr).into(), &rules).with_conf(conf);
        let _ = eqsat.run();
    }
}
