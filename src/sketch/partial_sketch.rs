// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Display, Formatter};
use std::mem::{discriminant, Discriminant};

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{Display, EnumDiscriminants, EnumIter, IntoEnumIterator, IntoStaticStr, VariantArray};

use super::{SketchLang, SketchParseError};
use crate::trs::{MetaInfo, SymbolType};
// use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type PartialSketch<L> = RecExpr<PartialSketchLang<L>>;

#[derive(
    Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize, EnumDiscriminants,
)]
#[strum_discriminants(derive(EnumIter, Display, VariantArray, IntoStaticStr))]
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
    type EnumDiscriminant = PartialSketchLangDiscriminants;
    const NON_OPERATORS: &'static [Self::EnumDiscriminant] =
        &[PartialSketchLangDiscriminants::Finished];

    fn symbol_type(&self) -> SymbolType {
        match self {
            PartialSketchLang::Finished(l) => l.symbol_type(),
            PartialSketchLang::Open => SymbolType::MetaSymbol(1 + L::operator_names().len()),
            PartialSketchLang::Active => SymbolType::MetaSymbol(2 + L::operator_names().len()),
        }
    }

    fn operator_id(&self) -> Option<usize> {
        match self {
            PartialSketchLang::Finished(l) => l.operator_id(),
            PartialSketchLang::Open | PartialSketchLang::Active => Self::EnumDiscriminant::iter()
                .filter(|x| !Self::NON_OPERATORS.contains(x))
                .position(|x| x == self.into())
                .map(|x| x + SketchLang::<L>::n_operators()),
        }
    }

    fn n_operators() -> usize {
        Self::EnumDiscriminant::VARIANTS.len() - Self::NON_OPERATORS.len()
            + SketchLang::<L>::n_operators()
    }

    fn operator_names() -> Vec<&'static str> {
        let outer_ops = Self::EnumDiscriminant::iter()
            .filter(|x| !Self::NON_OPERATORS.contains(x))
            .map(std::convert::Into::<&str>::into);
        let mut operators = SketchLang::<L>::operator_names();
        operators.extend(outer_ops);

        operators
    }
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
            "Add", "Sub", "Mul", "Div", "Mod", "Max", "Min", "Lt", "Gt", "Not", "Let", "Get", "Eq",
            "IEq", "Or", "And", "Any", "Contains", "Or", "Open", "Active",
        ];
        assert_eq!(
            PartialSketchLang::<HalideLang>::operator_names(),
            known_operators
        );
    }

    #[test]
    fn operator_id_partial_sketch() {
        let operator: PartialSketchLang<HalideLang> = PartialSketchLang::Open;
        assert_eq!(operator.operator_id(), Some(19));
    }

    #[test]
    fn operator_id_sketch() {
        let operator: PartialSketchLang<HalideLang> = PartialSketchLang::Finished(SketchLang::Any);
        assert_eq!(operator.operator_id(), Some(16));
    }

    #[test]
    fn operator_id_base() {
        let operator: PartialSketchLang<HalideLang> =
            PartialSketchLang::Finished(SketchLang::Node(HalideLang::Add([0.into(), 0.into()])));
        assert_eq!(operator.operator_id(), Some(0));
    }

    #[test]
    fn operator_id_base2() {
        let operator: PartialSketchLang<HalideLang> =
            PartialSketchLang::Finished(SketchLang::Node(HalideLang::Max([0.into(), 0.into()])));
        assert_eq!(operator.operator_id(), Some(5));
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
