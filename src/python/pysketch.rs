use std::fmt;
use std::fmt::{Display, Formatter};

use egg::{Language, RecExpr};
use pyo3::prelude::*;

use crate::sketches::SketchNode;

use super::macros::pyboxable;

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub enum PySketch {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any {},
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node { s: String, children: Vec<PySketch> },
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains { s: Box<PySketch> },
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or { ss: Vec<PySketch> },
}

pyboxable!(PySketch);

impl Display for PySketch {
    #[allow(clippy::redundant_closure_for_method_calls)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PySketch::Any {} => write!(f, "?"),
            PySketch::Contains { s } => {
                write!(f, "(contains {s})")
            }
            PySketch::Or { ss } => {
                let inner = ss
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                write!(f, "(or {inner})")
            }
            PySketch::Node { s, children } => {
                if children.is_empty() {
                    write!(f, "{s}")
                } else {
                    let inner = children
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    write!(f, "({s} {inner})")
                }
            }
        }
    }
}

impl<L: Language + Display> From<&RecExpr<SketchNode<L>>> for PySketch {
    fn from(rec_expr: &RecExpr<SketchNode<L>>) -> Self {
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = rec_expr.as_ref().last().unwrap();
        parse_rec_expr_rec(root, rec_expr)
    }
}

fn parse_rec_expr_rec<L: Language + Display>(
    node: &SketchNode<L>,
    rec_expr: &RecExpr<SketchNode<L>>,
) -> PySketch {
    match node {
        SketchNode::Any => PySketch::Any {},
        SketchNode::Node(node) => PySketch::Node {
            s: node.to_string(),
            children: node
                .children()
                .iter()
                .map(|child_id| parse_rec_expr_rec(&rec_expr[*child_id], rec_expr))
                .collect(),
        },
        SketchNode::Contains(id) => PySketch::Contains {
            s: parse_rec_expr_rec(&rec_expr[*id], rec_expr).into(),
        },
        SketchNode::Or(ids) => PySketch::Or {
            ss: ids
                .iter()
                .map(|id| parse_rec_expr_rec(&rec_expr[*id], rec_expr))
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{RecExpr, SymbolLang};

    #[test]
    fn parse_and_print_contains() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<RecExpr<SketchNode<SymbolLang>>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }

    #[test]
    fn parse_and_print_or() {
        let string = "(or (f ?))";
        let sketch = string.parse::<RecExpr<SketchNode<SymbolLang>>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }

    #[test]
    fn parse_and_print_complex() {
        let string = "(or (g ?) (f (or (f ?) a)))";
        let sketch = string.parse::<RecExpr<SketchNode<SymbolLang>>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }
}
