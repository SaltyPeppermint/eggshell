mod rules;

// use std::cmp::Ordering;
// use std::fmt::Display;

use egg::{define_language, Analysis, DidMerge, Id, PatternAst, Subst, Symbol};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use strum::{EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{MetaInfo, SymbolType, TermRewriteSystem};
// use crate::typing::{Type, Typeable, TypingInfo};

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

pub type Constant = NotNan<f64>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize, Deserialize, EnumDiscriminants)]
    #[strum_discriminants(derive(EnumIter))]
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

impl MetaInfo for Math {
    fn symbol_type(&self) -> SymbolType {
        match self {
            Math::Symbol(name) => SymbolType::Variable(name.as_str()),
            Math::Constant(value) => SymbolType::Constant(0, (*value).into()),
            _ => {
                let position = MathDiscriminants::iter()
                    .position(|x| x == self.into())
                    .unwrap();
                SymbolType::Operator(position + Self::N_CONST_TYPES)
            }
        }
    }

    fn operator_names() -> Vec<&'static str> {
        vec![
            "d", "i", "+", "-", "*", "/", "pow", "ln", "sqrt", "sin", "cos",
        ]
    }

    const N_CONST_TYPES: usize = 1;

    // fn operators() -> Vec<&'static Self::EnumDiscriminant> {
    //     let mut o = MathDiscriminants::VARIANTS.to_vec();
    //     o.truncate(o.len() - 2);
    //     o
    // }
}

// impl Typeable for Math {
//     type Type = ArithmaticType;

//     fn type_info(&self) -> TypingInfo<Self::Type> {
//         TypingInfo::new(Self::Type::Top, Self::Type::Top)
//     }
// }

// #[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
// pub enum ArithmaticType {
//     Top,
//     Bottom,
// }

// impl Display for ArithmaticType {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Self::Top => write!(f, "Top (Float)"),
//             Self::Bottom => write!(f, "Bottom"),
//         }
//     }
// }

// impl PartialOrd for ArithmaticType {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         match (self, other) {
//             // Bottom Type is smaller than everything else
//             (Self::Top, Self::Bottom) => Some(Ordering::Greater),
//             (Self::Bottom, Self::Top) => Some(Ordering::Less),
//             (Self::Bottom, Self::Bottom) | (Self::Top, Self::Top) => Some(Ordering::Equal),
//         }
//     }
// }

// impl Type for ArithmaticType {
//     fn top() -> Self {
//         Self::Top
//     }

//     fn bottom() -> Self {
//         Self::Bottom
//     }
// }

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct ConstantFold;

impl Analysis<Math> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Math>)>;

    fn make(egraph: &mut EGraph, enode: &Math) -> Self::Data {
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

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Arithmetic;

impl TermRewriteSystem for Arithmetic {
    type Language = Math;
    type Analysis = ConstantFold;

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        self::rules::rules()
    }
}
