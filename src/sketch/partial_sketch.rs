// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Display, Formatter};
use std::mem::{discriminant, Discriminant};

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{SketchLang, SketchParseError};
use crate::trs::{MetaInfo, SymbolType};
// use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type PartialSketch<L> = RecExpr<PartialSketchLang<L>>;

#[derive(
    Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, EnumDiscriminants,
)]
#[strum_discriminants(derive(EnumIter))]
pub enum PartialSketchLang<L: Language> {
    /// Inactive open placeholder that needs to be filled
    Open,
    /// Open placeholder that is currently being worked on
    Active,
    /// Finshed parts represented by [`SketchNode`]
    Finished(SketchLang<L>),
}

impl<L: Language> Language for PartialSketchLang<L> {
    type Discriminant = (
        Discriminant<Self>,
        Option<<SketchLang<L> as Language>::Discriminant>,
    );

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialSketchLang::Finished(x) => (discr, Some(x.discriminant())),
            _ => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Open | Self::Active => &[],
            Self::Finished(n) => n.children(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Open | Self::Active => &mut [],
            Self::Finished(n) => n.children_mut(),
        }
    }
}

// impl<L: Typeable> Typeable for PartialSketchNode<L> {
//     type Type = L::Type;

//     fn type_info(&self) -> crate::typing::TypingInfo<Self::Type> {
//         match self {
//             Self::Open | Self::Active => {
//                 TypingInfo::new(Self::Type::top(), Self::Type::top()).infer_return_type()
//             }
//             Self::Finished(t) => t.type_info(),
//         }
//     }
// }

impl<L: Language + MetaInfo> MetaInfo for PartialSketchLang<L> {
    fn symbol_type(&self) -> SymbolType {
        if let PartialSketchLang::Finished(l) = self {
            l.symbol_type()
        } else {
            let position = PartialSketchLangDiscriminants::iter()
                .position(|x| x == self.into())
                .unwrap();
            SymbolType::Operator(position + Self::N_CONST_TYPES)
        }
    }

    fn named_symbols() -> Vec<&'static str> {
        let mut operators = SketchLang::<L>::named_symbols();
        operators.extend(vec!["[open]", "[active]"]);
        operators
    }

    const N_CONST_TYPES: usize = SketchLang::<L>::N_CONST_TYPES;

    // const N_META_TYPES: usize = SketchLang::<L>::N_META_TYPES + 2;
}

impl<L: Language + Display> Display for PartialSketchLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "[open]"),
            Self::Active => write!(f, "[active]"),
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
        }
    }
}

impl<L> egg::FromOp for PartialSketchLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "[open]" | "[OPEN]" | "[Open]" => {
                if children.is_empty() {
                    Ok(Self::Open)
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "[active]" | "[ACTIVE]" | "[Active]" => {
                if children.is_empty() {
                    Ok(Self::Active)
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            _ => SketchLang::from_op(op, children).map(Self::Finished),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trs::{halide::HalideLang, MetaInfo};

    // use crate::typing::typecheck_expr;

    use super::*;

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 [active]) ?))"
            .parse::<PartialSketch<HalideLang>>()
            .unwrap();
        assert_eq!(&sketch.to_string(), "(contains (max (min 1 [active]) ?))");
    }

    #[test]
    fn operators() {
        let known_operators = vec![
            "+", "-", "*", "/", "%", "max", "min", "<", ">", "!", "<=", ">=", "==", "!=", "||",
            "&&", "?", "contains", "or", "[open]", "[active]",
        ];
        assert_eq!(
            PartialSketchLang::<HalideLang>::named_symbols(),
            known_operators
        );
    }

    // #[test]
    // fn typecheck_partial_sketch2() {
    //     let sketch = "(or (< (min 1 [active]) ?) (== 2 ?))"
    //         .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
    //         .unwrap();

    //     // let type_map = collect_expr_types(&sketch).unwrap();
    //     // let graph = dot_typed_ast(Id::from(sketch.as_ref().len() - 1), &sketch, &type_map);
    //     // let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
    //     // println!("{dot:?}");

    //     assert!(typecheck_expr(&sketch).is_ok());
    // }

    // #[test]
    // fn typecheck_partial_sketch3() {
    //     let sketch = "(or (or (> 1 [active]) ?) (or 2 ?))"
    //         .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
    //         .unwrap();

    //     assert!(typecheck_expr(&sketch).is_err());
    // }
}
