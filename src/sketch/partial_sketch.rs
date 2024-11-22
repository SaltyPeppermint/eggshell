// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{self, Display, Formatter};
use std::mem::Discriminant;

use egg::{Id, Language, RecExpr};
use serde::Serialize;

use super::{SketchNode, SketchParseError};
use crate::python::RawAst;
use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type PartialSketch<L> = RecExpr<PartialSketchNode<L>>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize)]
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

impl<L: Language + Display> Display for PartialSketchNode<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "[open]"),
            Self::Active => write!(f, "[active]"),
            Self::Finished(node) => write!(f, "{node}"),
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

impl<L> TryFrom<&RawAst> for (Id, PartialSketch<L>)
where
    L::Error: Display,
    L: Language + egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn try_from(pysketch: &RawAst) -> Result<Self, Self::Error> {
        fn rec<L>(
            sketch: &mut PartialSketch<L>,
            raw_sketch: &RawAst,
        ) -> Result<Id, SketchParseError<L::Error>>
        where
            L::Error: Display,
            L: Language + egg::FromOp,
        {
            // The recursions does terminate with either `PySketch::Any` or a childless
            // `PySketch::Node` since a sketch contains no back edges and is finite
            match raw_sketch {
                // No recursion here
                RawAst::Any => {
                    let id = sketch.add(PartialSketchNode::Finished(SketchNode::Any));
                    Ok(id)
                }
                RawAst::Open => {
                    let id = sketch.add(PartialSketchNode::Open);
                    Ok(id)
                }
                RawAst::Active => {
                    let id = sketch.add(PartialSketchNode::Active);
                    Ok(id)
                }
                RawAst::Node {
                    lang_node: s,
                    children,
                    features: _,
                } => {
                    // The recursions operate on a stricktly smaller PySketch with less elements in it.
                    // If this node contains no children, the child_ids will be an empty vector and this
                    // is the end of one of the recusions
                    let child_ids = children
                        .iter()
                        .map(|child| rec(sketch, child))
                        .collect::<Result<_, _>>()?;
                    let node = L::from_op(s, child_ids).map_err(SketchParseError::BadOp)?;
                    let id = sketch.add(PartialSketchNode::Finished(SketchNode::Node(node)));
                    Ok(id)
                }
                RawAst::Contains(node) => {
                    // Recursion reduces the number of the remaining elements in the PySketch by removing
                    // the wrapping `PySketch::Contains`
                    let child_id = rec(sketch, node)?;
                    let id =
                        sketch.add(PartialSketchNode::Finished(SketchNode::Contains(child_id)));
                    Ok(id)
                }
                RawAst::Or(children) => {
                    // Recursions reduces the number of the remaining elements in the PySketch since the or is removed
                    let child_0 = rec(sketch, &children[0])?;
                    let child_1 = rec(sketch, &children[1])?;

                    let id = sketch.add(PartialSketchNode::Finished(SketchNode::Or([
                        child_0, child_1,
                    ])));
                    Ok(id)
                }
            }
        }
        let mut sketch: PartialSketch<L> = RecExpr::default();
        let root_id = rec(&mut sketch, pysketch)?;
        Ok((root_id, sketch))
    }
}

impl<L> TryFrom<&RawAst> for PartialSketch<L>
where
    L::Error: Display,
    L: Language + egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn try_from(pysketch: &RawAst) -> Result<Self, Self::Error> {
        let (_, sketch) = pysketch.try_into()?;
        Ok(sketch)
    }
}

#[cfg(test)]
mod tests {
    use crate::sketch;
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
