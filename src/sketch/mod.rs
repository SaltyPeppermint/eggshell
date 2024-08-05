// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

mod analysis;
pub mod extract;
mod hashcons;
mod recursive;
mod utils;

use std::fmt::{self, Display, Formatter};

use egg::{Id, Language, RecExpr};
use serde::Serialize;

use crate::errors::SketchParseError;
use crate::python::PySketch;

/// Simple alias
pub type Sketch<L> = RecExpr<SketchNode<L>>;

#[derive(Debug, Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Serialize)]
pub(crate) enum SketchNode<L: Language> {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any,
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node(L),
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains(Id),
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or(Vec<Id>),
}

impl<L: Language> Language for SketchNode<L> {
    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Any => &[],
            Self::Node(n) => n.children(),
            Self::Contains(s) => std::slice::from_ref(s),
            Self::Or(ss) => ss.as_slice(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Any => &mut [],
            Self::Node(n) => n.children_mut(),
            Self::Contains(s) => std::slice::from_mut(s),
            Self::Or(ss) => ss.as_mut_slice(),
        }
    }
}

impl<L: Language + Display> Display for SketchNode<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => write!(f, "?"),
            Self::Node(node) => write!(f, "{node}"),
            Self::Contains(_) => write!(f, "contains"),
            Self::Or(_) => write!(f, "or"),
        }
    }
}

impl<L> egg::FromOp for SketchNode<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "?" => {
                if children.is_empty() {
                    Ok(Self::Any)
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "contains" => {
                if children.len() == 1 {
                    Ok(Self::Contains(children[0]))
                } else {
                    Err(SketchParseError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }
            "or" => Ok(Self::Or(children)),
            _ => L::from_op(op, children)
                .map(Self::Node)
                .map_err(SketchParseError::BadOp),
        }
    }
}

impl<L> TryFrom<&PySketch> for (Id, Sketch<L>)
where
    L::Error: Display,
    L: Language + egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn try_from(pysketch: &PySketch) -> Result<Self, Self::Error> {
        fn rec_try_from<L>(
            rec_expr: &mut Sketch<L>,
            pysketch: &PySketch,
        ) -> Result<Id, SketchParseError<L::Error>>
        where
            L::Error: Display,
            L: Language + egg::FromOp,
        {
            // The recursions does terminate with either `PySketch::Any` or a childless
            // `PySketch::Node` since a sketch contains no back edges and is finite
            match pysketch {
                // No recursion here
                PySketch::Any {} => {
                    let id = rec_expr.add(SketchNode::Any);
                    Ok(id)
                }
                PySketch::Node { s, children } => {
                    // The recursions operate on a stricktly smaller PySketch with less elements in it.
                    // If this node contains no children, the child_ids will be an empty vector and this
                    // is the end of one of the recusions
                    let child_ids = children
                        .iter()
                        .map(|child| rec_try_from(rec_expr, child))
                        .collect::<Result<_, _>>()?;
                    let node = L::from_op(s, child_ids).map_err(SketchParseError::BadOp)?;
                    let id = rec_expr.add(SketchNode::Node(node));
                    Ok(id)
                }

                PySketch::Contains { s } => {
                    // Recursion reduces the number of the remaining elements in the PySketch by removing
                    // the wrapping `PySketch::Contains`
                    let id = rec_try_from(rec_expr, s)?;
                    let id = rec_expr.add(SketchNode::Contains(id));
                    Ok(id)
                }
                PySketch::Or { ss } => {
                    // Recursions reduces the number of the remaining elements in the PySketch since the or is removed
                    let ids = ss
                        .iter()
                        .map(|child| rec_try_from(rec_expr, child))
                        .collect::<Result<_, _>>()?;
                    let id = rec_expr.add(SketchNode::Or(ids));
                    Ok(id)
                }
            }
        }
        let mut rec_expr: Sketch<L> = "".parse().unwrap();
        let root_id = rec_try_from(&mut rec_expr, pysketch)?;
        Ok((root_id, rec_expr))
    }
}

impl<L> TryFrom<&PySketch> for Sketch<L>
where
    L::Error: Display,
    L: Language + egg::FromOp,
{
    type Error = SketchParseError<L::Error>;

    fn try_from(value: &PySketch) -> Result<Self, Self::Error> {
        let (_, rec_expr) = value.try_into()?;
        Ok(rec_expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{RecExpr, SymbolLang};

    #[test]
    fn parse_and_print() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();

        let mut sketch_ref = RecExpr::default();
        let any = sketch_ref.add(SketchNode::Any);
        let f = sketch_ref.add(SketchNode::Node(SymbolLang::new("f", vec![any])));
        let _ = sketch_ref.add(SketchNode::Contains(f));

        assert_eq!(sketch, sketch_ref);
        assert_eq!(sketch.to_string(), string);
    }
}
