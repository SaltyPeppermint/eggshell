use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{Language, RecExpr};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyLang {
    s: String,
    children: Vec<PyLang>,
}

#[pymethods]
impl PyLang {
    #[new]
    pub fn new(s: String, children: Vec<PyLang>) -> Self {
        PyLang { s, children }
    }

    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let py_sketch = s_expr_str.parse()?;
        Ok(py_sketch)
    }

    pub fn stringify(&self) -> String {
        self.to_string()
    }
}

impl Display for PyLang {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.children.is_empty() {
            write!(f, "{}", self.s)
        } else {
            let inner = self
                .children
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "({} {})", self.s, inner)
        }
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PyLang`].
#[derive(Debug, Error)]
pub enum PyLangParseError {
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
    PyLangParseException,
    PyException,
    "Error parsing a PyLang."
);

impl From<PyLangParseError> for PyErr {
    fn from(err: PyLangParseError) -> PyErr {
        PyLangParseException::new_err(err.to_string())
    }
}

impl FromStr for PyLang {
    type Err = PyLangParseError;

    fn from_str(s: &str) -> Result<Self, PyLangParseError> {
        fn rec(sexp: &Sexp) -> Result<PyLang, PyLangParseError> {
            match sexp {
                Sexp::Empty => Err(PyLangParseError::EmptySexp),
                Sexp::String(s) => Ok(PyLang {
                    s: s.to_owned(),
                    children: vec![],
                }),
                Sexp::List(list) if list.is_empty() => Err(PyLangParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(PyLangParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => {
                        let children: Vec<PyLang> =
                            list[1..].iter().map(rec).collect::<Result<_, _>>()?;
                        Ok(PyLang {
                            s: s.to_owned(),
                            children,
                        })
                    }
                },
            }
        }

        let sexp =
            symbolic_expressions::parser::parse_str(s.trim()).map_err(PyLangParseError::BadSexp)?;
        rec(&sexp)
    }
}

impl<L: Language + Display> From<&RecExpr<L>> for PyLang {
    fn from(expr: &RecExpr<L>) -> Self {
        fn rec<L: Language + Display>(node: &L, expr: &RecExpr<L>) -> PyLang {
            PyLang {
                s: node.to_string(),
                children: node
                    .children()
                    .iter()
                    .map(|child_id| {
                        let child = &expr[*child_id];
                        rec(child, expr)
                    })
                    .collect(),
            }
        }
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = expr.as_ref().last().unwrap();
        rec(root, expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trs::halide::HalideMath;

    #[test]
    fn parse_basic() {
        let lhs = PyLang {
            s: "==".to_owned(),
            children: vec![
                PyLang {
                    s: "0".to_owned(),
                    children: vec![],
                },
                PyLang {
                    s: "0".to_owned(),
                    children: vec![],
                },
            ],
        };
        let rhs: RecExpr<HalideMath> = "(== 0 0)".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }

    #[test]
    fn print_basic() {
        let lhs = PyLang {
            s: "==".to_owned(),
            children: vec![
                PyLang {
                    s: "0".to_owned(),
                    children: vec![],
                },
                PyLang {
                    s: "0".to_owned(),
                    children: vec![],
                },
            ],
        }
        .to_string();
        let rhs = "(== 0 0)";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_nested() {
        let lhs = PyLang {
            s: "==".to_owned(),

            children: vec![
                PyLang {
                    s: "+".to_owned(),
                    children: vec![
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    s: "2".to_owned(),
                    children: vec![],
                },
            ],
        };
        let rhs: RecExpr<HalideMath> = "(== (+ 1 1) 2)".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }

    #[test]
    fn print_nested() {
        let lhs = PyLang {
            s: "==".to_owned(),

            children: vec![
                PyLang {
                    s: "+".to_owned(),
                    children: vec![
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    s: "2".to_owned(),
                    children: vec![],
                },
            ],
        }
        .to_string();
        let rhs = "(== (+ 1 1) 2)";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_complex() {
        let lhs = PyLang {
            s: "==".to_owned(),
            children: vec![
                PyLang {
                    s: "+".to_owned(),
                    children: vec![
                        PyLang {
                            s: "+".to_owned(),
                            children: vec![
                                PyLang {
                                    s: "1".to_owned(),
                                    children: vec![],
                                },
                                PyLang {
                                    s: "0".to_owned(),
                                    children: vec![],
                                },
                            ],
                        },
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    s: "+".to_owned(),
                    children: vec![
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            s: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
            ],
        };
        let rhs: RecExpr<HalideMath> = "(== (+ (+ 1 0) 1) (+ 1 1))".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }
}
