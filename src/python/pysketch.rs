use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::Language;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use crate::sketch::{Sketch, SketchNode};

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
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to a single leaf node in the underlying language syntax.
    Leaf { s: String },
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains { s: Box<PySketch> },
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or { ss: Vec<PySketch> },
}

#[pymethods]
impl PySketch {
    #[new]
    pub fn new(s_expr_str: &str) -> PyResult<Self> {
        let py_sketch = s_expr_str.parse()?;
        Ok(py_sketch)
    }
}

pyboxable!(PySketch);

impl Display for PySketch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PySketch::Any {} => write!(f, "?"),
            PySketch::Contains { s } => write!(f, "(contains {s})"),
            PySketch::Or { ss } => {
                let inner = ss
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                write!(f, "(or {inner})")
            }
            PySketch::Leaf { s } => write!(f, "{s}"),
            PySketch::Node { s, children } => {
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

/// An error type for failures when attempting to parse an s-expression as a
/// [`RecExpr<L>`].
#[derive(Debug, Error)]
pub enum PySketchParseError {
    /// An empty s-expression was found. Usually this is caused by an
    /// empty list "()" somewhere in the input.
    #[error("found empty s-expression")]
    EmptySexp,

    /// A list was found where an operator was expected. This is caused by
    /// s-expressions of the form "((a b c) d e f)."
    #[error("found a list in the head position: {0}")]
    HeadList(Sexp),

    /// An error occurred while parsing the s-expression itself, generally
    /// because the input had an invalid structure (e.g. unpaired parentheses).
    #[error(transparent)]
    BadSexp(SexpError),
}

create_exception!(
    eggshell,
    PySketchParseException,
    PyException,
    "Error parsing a PySketch."
);

impl From<PySketchParseError> for PyErr {
    fn from(err: PySketchParseError) -> PyErr {
        PySketchParseException::new_err(err.to_string())
    }
}

impl FromStr for PySketch {
    type Err = PySketchParseError;

    fn from_str(s: &str) -> Result<Self, PySketchParseError> {
        fn rec(sexp: &Sexp) -> Result<PySketch, PySketchParseError> {
            match sexp {
                Sexp::Empty => Err(PySketchParseError::EmptySexp),
                Sexp::String(s) => match s.as_str() {
                    "?" => Ok(PySketch::Any {}),
                    _ => Ok(PySketch::Leaf { s: s.to_owned() }),
                },
                Sexp::List(list) if list.is_empty() => Err(PySketchParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(PySketchParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(head) => {
                        if head.as_str() == "contains" {
                            let inner = rec(&list[1])?;
                            Ok(PySketch::Contains { s: Box::new(inner) })
                        } else {
                            let children: Vec<PySketch> =
                                list[1..].iter().map(rec).collect::<Result<_, _>>()?;
                            match head.as_str() {
                                "or" => Ok(PySketch::Or { ss: children }),
                                _ => Ok(PySketch::Node {
                                    s: head.to_owned(),
                                    children,
                                }),
                            }
                        }
                    }
                },
            }
        }

        let sexp = symbolic_expressions::parser::parse_str(s.trim())
            .map_err(PySketchParseError::BadSexp)?;
        rec(&sexp)
    }
}

impl<L: Language + Display> From<&Sketch<L>> for PySketch {
    fn from(sketch: &Sketch<L>) -> Self {
        fn rec<L: Language + Display>(node: &SketchNode<L>, sketch: &Sketch<L>) -> PySketch {
            match node {
                SketchNode::Any => PySketch::Any {},
                SketchNode::Node(inner_node) => {
                    if inner_node.is_leaf() {
                        PySketch::Leaf {
                            s: inner_node.to_string(),
                        }
                    } else {
                        PySketch::Node {
                            s: inner_node.to_string(),
                            children: inner_node
                                .children()
                                .iter()
                                .map(|child_id| rec(&sketch[*child_id], sketch))
                                .collect(),
                        }
                    }
                }
                SketchNode::Contains(id) => PySketch::Contains {
                    s: rec(&sketch[*id], sketch).into(),
                },
                SketchNode::Or(ids) => PySketch::Or {
                    ss: ids.iter().map(|id| rec(&sketch[*id], sketch)).collect(),
                },
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = sketch.as_ref().last().unwrap();
        rec(root, sketch)
    }
}

impl<L: Language + Display> From<Sketch<L>> for PySketch {
    fn from(sketch: Sketch<L>) -> Self {
        (&sketch).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::SymbolLang;

    #[test]
    fn parse_and_print_contains() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }

    #[test]
    fn parse_and_print_or() {
        let string = "(or (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }

    #[test]
    fn parse_and_print_complex() {
        let string = "(or (g ?) (f (or (f ?) a)))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), string);
    }

    #[test]
    fn parse_pysketch_vs_sketch() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = string.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn pysketch_constructor() {
        let string = "(contains (f ?))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = PySketch::new(string).unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn parse_pysketch_vs_sketch_complex() {
        let string = "(or (g ?) (f (or (f ?) a)))";
        let sketch = string.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = string.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }
}
