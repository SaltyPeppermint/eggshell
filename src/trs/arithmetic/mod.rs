mod rules;

use std::{cmp::Ordering, fmt::Display};

use egg::{define_language, Analysis, DidMerge, Id, PatternAst, Subst, Symbol};
use ordered_float::NotNan;
use serde::Serialize;

use super::{Ruleset, SymbolIter, Trs, TrsError};
use crate::typing::{Type, Typeable, TypingInfo};

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

impl SymbolIter for Math {
    fn raw_symbols() -> &'static [(&'static str, usize)] {
        &[
            ("d", 2),
            ("i", 2),
            ("+", 2),
            ("-", 2),
            ("*", 2),
            ("/", 2),
            ("pow", 2),
            ("ln", 1),
            ("sqrt", 1),
            ("sin", 1),
            ("cos", 1),
        ]
    }
}

impl Typeable for Math {
    type Type = ArithmaticType;

    fn type_info(&self) -> TypingInfo<Self::Type> {
        TypingInfo::new(Self::Type::Top, Self::Type::Top)
    }
}

// pub struct MathCostFn;
// impl egg::CostFunction<Math> for MathCostFn {
//     type Cost = usize;
//     fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost,
//     {
//         let op_cost = match enode {
//             Math::Diff(..) | Math::Integral(..) => 100,
//             _ => 1,
//         };
//         enode.fold(op_cost, |sum, i| sum + costs(i))
//     }
// }

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum ArithmaticType {
    Top,
    Bottom,
}

impl Display for ArithmaticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Top => write!(f, "Top (Float)"),
            Self::Bottom => write!(f, "Bottom"),
        }
    }
}

impl PartialOrd for ArithmaticType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Bottom Type is smaller than everything else
            (Self::Top, Self::Bottom) => Some(Ordering::Greater),
            (Self::Bottom, Self::Top) => Some(Ordering::Less),
            (Self::Bottom, Self::Bottom) | (Self::Top, Self::Top) => Some(Ordering::Equal),
        }
    }
}

impl Type for ArithmaticType {
    fn top() -> Self {
        Self::Top
    }

    fn bottom() -> Self {
        Self::Bottom
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

#[derive(Debug, Clone, Copy, Serialize)]
pub enum ArithmaticRules {
    Full,
}

impl Ruleset for ArithmaticRules {
    type Language = Math;
    type Analysis = ConstantFold;
    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[must_use]
    fn rules(&self) -> Vec<Rewrite> {
        self::rules::rules()
    }
}

impl TryFrom<String> for ArithmaticRules {
    type Error = TrsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            _ => Err(Self::Error::BadRulesetName(value)),
        }
    }
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Arithmetic;

impl Trs for Arithmetic {
    type Language = Math;
    type Analysis = ConstantFold;
    type Rules = ArithmaticRules;
}
