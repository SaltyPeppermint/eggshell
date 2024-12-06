// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Display, Formatter};
use std::mem::Discriminant;

use egg::{Id, Language, RecExpr};
use serde::{Deserialize, Serialize};

use super::{SketchNode, SketchParseError};
use crate::features::{AsFeatures, SymbolType};
use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type PartialSketch<L> = RecExpr<PartialSketchNode<L>>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PartialSketchNode<L: Language> {
    /// Inactive open placeholder that needs to be filled
    Open,
    /// Open placeholder that is currently being worked on
    Active,
    /// Finshed parts represented by [`SketchNode`]
    Finished(SketchNode<L>),
}

impl<L: Language> Language for PartialSketchNode<L> {
    type Discriminant = (Discriminant<Self>, Discriminant<SketchNode<L>>);

    fn discriminant(&self) -> Self::Discriminant {
        panic!("Comparing sketches to each other does not make sense!")
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

impl<L: Typeable> Typeable for PartialSketchNode<L> {
    type Type = L::Type;

    fn type_info(&self) -> crate::typing::TypingInfo<Self::Type> {
        match self {
            Self::Open | Self::Active => {
                TypingInfo::new(Self::Type::top(), Self::Type::top()).infer_return_type()
            }
            Self::Finished(t) => t.type_info(),
        }
    }
}

impl<L: Language + AsFeatures> AsFeatures for PartialSketchNode<L> {
    fn featurizer(variable_names: Vec<String>) -> crate::features::Featurizer<Self> {
        SketchNode::<L>::featurizer(variable_names)
            .into_meta_lang(|l| PartialSketchNode::Finished(l))
    }

    fn symbol_type(&self) -> SymbolType {
        match self {
            PartialSketchNode::Finished(l) => l.symbol_type(),
            _ => SymbolType::MetaSymbol,
        }
    }

    fn into_symbol(name: String) -> Self {
        PartialSketchNode::Finished(SketchNode::Node(L::into_symbol(name)))
    }
}

impl<L: Language + Display> Display for PartialSketchNode<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "[open]"),
            Self::Active => write!(f, "[active]"),
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
        }
    }
}

impl<L> egg::FromOp for PartialSketchNode<L>
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
            _ => SketchNode::from_op(op, children).map(Self::Finished),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trs::{Halide, TermRewriteSystem};
    use crate::typing::typecheck_expr;

    use super::*;

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 [active]) ?))"
            .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();
        assert_eq!(&sketch.to_string(), "(contains (max (min 1 [active]) ?))");
    }

    #[test]
    fn typecheck_partial_sketch1() {
        let sketch = "(or (max (min 1 [active]) ?) (== 2 ?))"
            .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();

        assert!(typecheck_expr(&sketch).is_err());
    }

    #[test]
    fn typecheck_partial_sketch2() {
        let sketch = "(or (< (min 1 [active]) ?) (== 2 ?))"
            .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();

        // let type_map = collect_expr_types(&sketch).unwrap();
        // let graph = dot_typed_ast(Id::from(sketch.as_ref().len() - 1), &sketch, &type_map);
        // let dot = Dot::with_config(&graph, &[Config::EdgeNoLabel]);
        // println!("{dot:?}");

        assert!(typecheck_expr(&sketch).is_ok());
    }

    #[test]
    fn typecheck_partial_sketch3() {
        let sketch = "(or (or (> 1 [active]) ?) (or 2 ?))"
            .parse::<PartialSketch<<Halide as TermRewriteSystem>::Language>>()
            .unwrap();

        assert!(typecheck_expr(&sketch).is_err());
    }
}
