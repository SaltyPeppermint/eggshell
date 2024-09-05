use std::fmt;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{FromOp, Id, Language, RecExpr};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use crate::{HashMap, HashSet};

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyLang {
    node: String,
    children: Vec<PyLang>,
}

#[pymethods]
impl PyLang {
    #[new]
    pub fn new(node: String, children: Vec<PyLang>) -> Self {
        PyLang { node, children }
    }

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

    /// Returns Hashmap of all the symbols present in it with their possible children
    pub fn symbols(&self) -> HashMap<String, HashSet<String>> {
        fn rec(pylang: &PyLang, children: &mut HashMap<String, HashSet<String>>) {
            if let Some(s) = children.get_mut(&pylang.node) {
                s.extend(pylang.children.iter().map(|c| c.node.clone()));
            } else {
                children.insert(
                    pylang.node.clone(),
                    pylang.children.iter().map(|c| c.node.clone()).collect(),
                );
            }
            for child in &pylang.children {
                rec(child, children);
            }
        }
        let mut children = HashMap::new();
        rec(self, &mut children);
        children
    }

    /// Returns Hashmap of all the symbols present in it with their possible children
    pub fn arity(&self) -> HashMap<String, HashSet<usize>> {
        fn rec(pylang: &PyLang, children: &mut HashMap<String, HashSet<usize>>) {
            if let Some(s) = children.get_mut(&pylang.node) {
                s.insert(pylang.children.len());
            } else {
                let mut s = HashSet::new();
                s.insert(pylang.children.len());
                children.insert(pylang.node.clone(), s);
            }
            for child in &pylang.children {
                rec(child, children);
            }
        }
        let mut children = HashMap::new();
        rec(self, &mut children);
        children
    }
}

impl Display for PyLang {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.children.is_empty() {
            write!(f, "{}", self.node)
        } else {
            let inner = self
                .children
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "({} {})", self.node, inner)
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
                    node: s.to_owned(),
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
                            node: s.to_owned(),
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
                node: node.to_string(),
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

impl<L> TryFrom<&PyLang> for RecExpr<L>
where
    L: Language + FromOp,
{
    type Error = <L as egg::FromOp>::Error;

    fn try_from(value: &PyLang) -> Result<Self, Self::Error> {
        fn rec<L: Language + FromOp>(
            pylang: &PyLang,
            rec_expr: &mut RecExpr<L>,
        ) -> Result<Id, <L as egg::FromOp>::Error> {
            if let 0 = pylang.children.len() {
                let node = L::from_op(&pylang.node, vec![])?;
                Ok(rec_expr.add(node))
            } else {
                let children = pylang
                    .children
                    .iter()
                    .map(|c| rec(c, rec_expr))
                    .collect::<Result<_, _>>()?;
                let node = L::from_op(&pylang.node, children)?;
                Ok(rec_expr.add(node))
            }
        }

        let mut rec_expr = RecExpr::<L>::default();
        rec(value, &mut rec_expr)?;
        Ok(rec_expr)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use super::*;
    use crate::trs::halide::HalideMath;

    #[test]
    fn convert_pylang() {
        let term = "(== (+ (+ 1 0) 1) (+ 1 1))";
        let lhs: RecExpr<HalideMath> = term.parse::<PyLang>().unwrap().borrow().try_into().unwrap();
        let rhs: RecExpr<HalideMath> = term.parse().unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn parse_basic() {
        let lhs = PyLang {
            node: "==".to_owned(),
            children: vec![
                PyLang {
                    node: "0".to_owned(),
                    children: vec![],
                },
                PyLang {
                    node: "0".to_owned(),
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
            node: "==".to_owned(),
            children: vec![
                PyLang {
                    node: "0".to_owned(),
                    children: vec![],
                },
                PyLang {
                    node: "0".to_owned(),
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
            node: "==".to_owned(),

            children: vec![
                PyLang {
                    node: "+".to_owned(),
                    children: vec![
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    node: "2".to_owned(),
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
            node: "==".to_owned(),

            children: vec![
                PyLang {
                    node: "+".to_owned(),
                    children: vec![
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    node: "2".to_owned(),
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
            node: "==".to_owned(),
            children: vec![
                PyLang {
                    node: "+".to_owned(),
                    children: vec![
                        PyLang {
                            node: "+".to_owned(),
                            children: vec![
                                PyLang {
                                    node: "1".to_owned(),
                                    children: vec![],
                                },
                                PyLang {
                                    node: "0".to_owned(),
                                    children: vec![],
                                },
                            ],
                        },
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                PyLang {
                    node: "+".to_owned(),
                    children: vec![
                        PyLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        PyLang {
                            node: "1".to_owned(),
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
