use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::Language;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use super::macros::pyboxable;
use crate::sketch::{PartialSketch, PartialSketchNode, Sketch, SketchNode};

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub enum PySketch {
    /// Any program of the underlying [`Language`].
    ///
    /// Corresponds to the `?` syntax.
    Any {},
    /// In case the sketch is unfinished, there are still open slots to be filled
    ///
    /// This is an inactive todo
    Todo {},
    /// In case the sketch is unfinished, there are still open slots to be filled
    ///
    /// This is an active todo being currently worked on
    Active {},
    /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    ///
    /// Corresponds to the `(language_node s1 .. sn)` syntax.
    Node {
        lang_node: String,
        children: Vec<PySketch>,
    },
    // /// Programs made from this [`Language`] node whose children satisfy the given sketches.
    // ///
    // /// Corresponds to a single leaf node in the underlying language syntax.
    // Leaf { s: String },
    /// Programs that contain sub-programs satisfying the given sketch.
    ///
    /// Corresponds to the `(contains s)` syntax.
    Contains { node: Box<PySketch> },
    /// Programs that satisfy any of these sketches.
    ///
    /// Corresponds to the `(or s1 .. sn)` syntax.
    Or { children: Vec<PySketch> },
}

pyboxable!(PySketch);

#[pymethods]
impl PySketch {
    #[new]
    #[pyo3(signature = (node_type, children=vec![]))]
    fn new(node_type: &str, mut children: Vec<PySketch>) -> PyResult<Self> {
        match (node_type, children.len()) {
            ("any" | "ANY" | "Any"| "?", 0) => Ok(PySketch::Any {}),
            ("[todo]" | "[TODO]" | "[Todo]", 0) => Ok(PySketch::Todo {}),
            ("[active]" | "[ACTIVE]" | "[Active]", 0) => Ok(PySketch::Active {}),
            ("any" | "ANY" | "Any"| "?", n) => {
                Err(PyErr::new::<PyValueError, _>(
                    format!("Any does not have any children. You supplied children: {n:?}"),
                ))
            }
            ("[todo]" | "[TODO]" | "[Todo]", n) => {
                Err(PyErr::new::<PyValueError, _>(
                    format!("'[Todo]' does not have any children. You supplied children: {n:?}"),
                ))
            }
            ("[active]" | "[ACTIVE]" | "[Active]", n) => {
                Err(PyErr::new::<PyValueError, _>(
                    format!("'[Active]' does not have any children. You supplied children: {n:?}"),
                ))
            }
            ("or" | "OR" | "Or", n @ (0 | 1))  => Err(PyErr::new::<PyValueError, _>(
                format!(
                    "'Or' must have more than 1 child, not {n} children. You supplied: {children:?}"
                ),
            )),
            ("or" | "OR" | "Or", _) => Ok(PySketch::Or { children }),
            ("contains" | "CONTAINS" | "Contains", 1) => Ok(PySketch::Contains {
                node: Box::new(children.swap_remove(0))
            }),
            ("contains" | "CONTAINS" | "Contains", n) => Err(PyErr::new::<PyValueError, _>(
                        format!(
                            "'Contains' can only have a 1 child, no more nor less, not {n} children. You supplied: {children:?}"
                        ),
                    )),

            (s, _) => Ok(PySketch::Node { lang_node: s.parse()?, children})
        }
    }

    /// You will probably want to use this
    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let py_sketch = s_expr_str.parse()?;
        Ok(py_sketch)
    }

    #[pyo3(name = "__str__")]
    pub fn stringify(&self) -> String {
        self.to_string()
    }

    #[pyo3(name = "__repr__")]
    pub fn debug(&self) -> String {
        format!("{self:?}")
    }
}

impl Display for PySketch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            PySketch::Any {} => write!(f, "?"),
            PySketch::Todo {} => write!(f, "[todo]"),
            PySketch::Active {} => write!(f, "[active]"),
            PySketch::Contains { node } => write!(f, "(contains {node})"),
            PySketch::Or { children } => {
                let inner = children
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                write!(f, "(or {inner})")
            }
            PySketch::Node {
                lang_node,
                children,
            } => {
                if children.is_empty() {
                    write!(f, "{lang_node}")
                } else {
                    let inner = children
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(" ");
                    write!(f, "({lang_node} {inner})")
                }
            }
        }
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PySketch`].
#[derive(Debug, Error)]
pub enum PySketchParseError {
    /// An empty s-expression was found. Usually this is caused by an
    /// empty list "()" somewhere in the input.
    #[error("Found empty s-expression")]
    EmptySexp,

    /// A list was found where an operator was expected. This is caused by
    /// s-expressions of the form "((a b c) d e f)."
    #[error("Found a list in the head position: {0}")]
    HeadList(Sexp),

    /// A or expression was found where with less than 2 children.
    #[error("Found an or with less than 2 children: {0:?}")]
    BadChildrenOr(Vec<Sexp>),

    /// A or expression was found where with less than 2 children.
    #[error("Found an or in a terminal position: {0}")]
    BadTerminalOr(Sexp),

    /// A contains expression was found where with more or less than 1 children.
    #[error("Found an 'contains' with more or less than 1 child: {0:?}")]
    BadChildrenContains(Vec<Sexp>),

    /// A or expression was found where with less than 2 children.
    #[error("Found an or in a terminal position: {0}")]
    BadTerminalContains(Sexp),

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
                    "?" | "any" | "ANY" | "Any" => Ok(PySketch::Any {}),
                    "[todo]" | "[TODO]" | "[Todo]" => Ok(PySketch::Todo {}),
                    "[active]" | "[ACTIVE]" | "[Active]" => Ok(PySketch::Active {}),
                    "or" | "OR" | "Or" => Err(PySketchParseError::BadTerminalOr(sexp.to_owned())),
                    "contains" | "CONTAINS" | "Contains" => {
                        Err(PySketchParseError::BadTerminalContains(sexp.to_owned()))
                    }
                    _ => Ok(PySketch::Node {
                        lang_node: s.to_string(),
                        children: vec![],
                    }),
                },
                Sexp::List(list) if list.is_empty() => Err(PySketchParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(PySketchParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => match (s.as_str(), list.len()) {
                        ("contains" | "CONTAINS" | "Contains", 2) => {
                            let inner = rec(&list[1])?;
                            Ok(PySketch::Contains {
                                node: Box::new(inner),
                            })
                        }
                        ("contains" | "CONTAINS" | "Contains", _) => {
                            Err(PySketchParseError::BadChildrenContains(list.to_owned()))
                        }
                        ("or" | "OR" | "Or", 0..=2) => {
                            Err(PySketchParseError::BadChildrenOr(list.to_owned()))
                        }
                        ("or" | "OR" | "Or", _) => Ok(PySketch::Or {
                            children: list[1..].iter().map(rec).collect::<Result<_, _>>()?,
                        }),
                        _ => Ok(PySketch::Node {
                            lang_node: s.to_owned(),
                            children: list[1..].iter().map(rec).collect::<Result<_, _>>()?,
                        }),
                    },
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
                SketchNode::Node(lang_node) => PySketch::Node {
                    lang_node: lang_node.to_string(),
                    children: lang_node
                        .children()
                        .iter()
                        .map(|child_id| rec(&sketch[*child_id], sketch))
                        .collect(),
                },
                SketchNode::Contains(id) => PySketch::Contains {
                    node: rec(&sketch[*id], sketch).into(),
                },
                SketchNode::Or(ids) => PySketch::Or {
                    children: ids.iter().map(|id| rec(&sketch[*id], sketch)).collect(),
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

impl<L: Language + Display> From<&PartialSketch<L>> for PySketch {
    fn from(sketch: &PartialSketch<L>) -> Self {
        fn rec<L: Language + Display>(
            node: &PartialSketchNode<L>,
            sketch: &PartialSketch<L>,
        ) -> PySketch {
            match node {
                PartialSketchNode::Any => PySketch::Any {},
                PartialSketchNode::Todo => PySketch::Todo {},
                PartialSketchNode::Active => PySketch::Active {},
                PartialSketchNode::Node(lang_node) => PySketch::Node {
                    lang_node: lang_node.to_string(),
                    children: lang_node
                        .children()
                        .iter()
                        .map(|child_id| rec(&sketch[*child_id], sketch))
                        .collect(),
                },
                PartialSketchNode::Contains(id) => PySketch::Contains {
                    node: rec(&sketch[*id], sketch).into(),
                },
                PartialSketchNode::Or(ids) => PySketch::Or {
                    children: ids.iter().map(|id| rec(&sketch[*id], sketch)).collect(),
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
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_or() {
        let expr = "(or f ?)";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_and_print_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch: PySketch = (&sketch).into();
        assert_eq!(pysketch.to_string(), expr);
    }

    #[test]
    fn parse_pysketch_vs_sketch() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn pysketch_from_str() {
        let expr = "(contains (f ?))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = PySketch::from_str(expr).unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn parse_pysketch_vs_sketch_complex() {
        let expr = "(or (g ?) (f (or (f ?) a)))";
        let sketch = expr.parse::<Sketch<SymbolLang>>().unwrap();
        let pysketch_a: PySketch = (&sketch).into();
        let pysketch_b = expr.parse().unwrap();
        assert_eq!(pysketch_a, pysketch_b);
    }

    #[test]
    fn bad_children_or() {
        let expr = "(or f)";
        let parse_error = expr.parse::<PySketch>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(PySketchParseError::BadChildrenOr(_))
        ));
    }

    #[test]
    fn bad_terminal_or() {
        let expr = "(f or)";
        let parse_error = expr.parse::<PySketch>();
        eprintln!("{parse_error:?}");
        assert!(matches!(
            parse_error,
            Err(PySketchParseError::BadTerminalOr(_))
        ));
    }

    #[test]
    fn bad_children_contains() {
        let expr = "(contains f g)";
        let parse_error = expr.parse::<PySketch>();
        assert!(matches!(
            parse_error,
            Err(PySketchParseError::BadChildrenContains(_))
        ));
    }

    #[test]
    fn bad_terminal_contains() {
        let expr = "(f contains)";
        let parse_error = expr.parse::<PySketch>();
        assert!(matches!(
            parse_error,
            Err(PySketchParseError::BadTerminalContains(_))
        ));
    }
}
