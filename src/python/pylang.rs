use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{FromOp, Id};
use egg::{Language, RecExpr};
use pyo3::exceptions::PyException;
use pyo3::{create_exception, prelude::*};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use super::FlatAst;
use crate::utils::Tree;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
/// Wrapper type for Python
pub struct PyLang(pub(crate) RawLang);

#[pymethods]
impl PyLang {
    #[new]
    pub fn new(node: String, children: Vec<PyLang>) -> Self {
        let raw_children = children.into_iter().map(|c| c.0).collect();
        let raw_lang = RawLang::new(node, raw_children);
        PyLang(raw_lang)
    }

    /// Parse from string
    #[staticmethod]
    pub fn from_str(s_expr_str: &str) -> PyResult<Self> {
        let raw_lang = s_expr_str.parse()?;
        Ok(PyLang(raw_lang))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    /// Returns a flat representation of itself
    pub fn flat(&self) -> FlatAst {
        (&self.0).into()
    }

    /// Returns the number of nodes in the sketch
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Returns the maximum AST depth in the sketch
    pub fn depth(&self) -> usize {
        self.0.depth()
    }
}

impl From<RawLang> for PyLang {
    fn from(value: RawLang) -> Self {
        PyLang(value)
    }
}

impl FromStr for PyLang {
    type Err = RawLangParseError;

    fn from_str(s: &str) -> Result<Self, RawLangParseError> {
        let raw_lang = s.parse()?;
        Ok(PyLang(raw_lang))
    }
}

impl<L: Language + Display> From<&RecExpr<L>> for PyLang {
    fn from(value: &RecExpr<L>) -> Self {
        let raw_lang = value.into();
        PyLang(raw_lang)
    }
}

create_exception!(
    eggshell,
    PyLangException,
    PyException,
    "Error parsing a PyLang."
);

impl From<RawLangParseError> for PyErr {
    fn from(err: RawLangParseError) -> PyErr {
        PyLangException::new_err(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawLang {
    node: String,
    children: Box<[RawLang]>,
}

impl RawLang {
    pub fn new(node: String, children: Vec<RawLang>) -> Self {
        RawLang {
            node,
            children: children.into_boxed_slice(),
        }
    }

    pub fn name(&self) -> &str {
        &self.node
    }
}

impl Tree for RawLang {
    fn children(&self) -> &[Self] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Self] {
        &mut self.children
    }
}

impl Display for RawLang {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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

impl FromStr for RawLang {
    type Err = RawLangParseError;

    fn from_str(s: &str) -> Result<Self, RawLangParseError> {
        fn rec(sexp: &Sexp) -> Result<RawLang, RawLangParseError> {
            match sexp {
                Sexp::Empty => Err(RawLangParseError::EmptySexp),
                Sexp::String(s) => Ok(RawLang {
                    node: s.to_owned(),
                    children: Box::new([]),
                }),
                Sexp::List(list) if list.is_empty() => Err(RawLangParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(RawLangParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => {
                        let children = list[1..].iter().map(rec).collect::<Result<_, _>>()?;
                        Ok(RawLang {
                            node: s.to_owned(),
                            children,
                        })
                    }
                },
            }
        }

        let sexp = symbolic_expressions::parser::parse_str(s.trim())
            .map_err(RawLangParseError::BadSexp)?;
        rec(&sexp)
    }
}

/// An error type for failures when attempting to parse an s-expression as a
/// [`PyLang`].
#[derive(Debug, Error)]
pub enum RawLangParseError {
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

impl<L: Language + Display> From<&RecExpr<L>> for RawLang {
    fn from(expr: &RecExpr<L>) -> Self {
        fn rec<L: Language + Display>(node: &L, expr: &RecExpr<L>) -> RawLang {
            RawLang {
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

impl<L> TryFrom<&RawLang> for RecExpr<L>
where
    L: Language + FromOp,
{
    type Error = <L as egg::FromOp>::Error;

    fn try_from(value: &RawLang) -> Result<Self, Self::Error> {
        fn rec<L: Language + FromOp>(
            pylang: &RawLang,
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
    use super::*;
    use crate::trs::{Halide, Trs};

    #[test]
    fn convert_pylang() {
        let term = "(== (+ (+ 1 0) 1) (+ 1 1))";
        let lhs: RecExpr<<Halide as Trs>::Language> =
            (&term.parse::<RawLang>().unwrap()).try_into().unwrap();
        let rhs: RecExpr<<Halide as Trs>::Language> = term.parse().unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn parse_basic() {
        let lhs = RawLang {
            node: "==".to_owned(),
            children: Box::new([
                RawLang {
                    node: "0".to_owned(),
                    children: Box::new([]),
                },
                RawLang {
                    node: "0".to_owned(),
                    children: Box::new([]),
                },
            ]),
        };
        let rhs: RecExpr<<Halide as Trs>::Language> = "(== 0 0)".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }

    #[test]
    fn print_basic() {
        let lhs = RawLang {
            node: "==".to_owned(),
            children: Box::new([
                RawLang {
                    node: "0".to_owned(),
                    children: Box::new([]),
                },
                RawLang {
                    node: "0".to_owned(),
                    children: Box::new([]),
                },
            ]),
        }
        .to_string();
        let rhs = "(== 0 0)";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_nested() {
        let lhs = RawLang {
            node: "==".to_owned(),

            children: Box::new([
                RawLang {
                    node: "+".to_owned(),
                    children: Box::new([
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                    ]),
                },
                RawLang {
                    node: "2".to_owned(),
                    children: Box::new([]),
                },
            ]),
        };
        let rhs: RecExpr<<Halide as Trs>::Language> = "(== (+ 1 1) 2)".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }

    #[test]
    fn print_nested() {
        let lhs = RawLang {
            node: "==".to_owned(),

            children: Box::new([
                RawLang {
                    node: "+".to_owned(),
                    children: Box::new([
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                    ]),
                },
                RawLang {
                    node: "2".to_owned(),
                    children: Box::new([]),
                },
            ]),
        }
        .to_string();
        let rhs = "(== (+ 1 1) 2)";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_complex() {
        let lhs = RawLang {
            node: "==".to_owned(),
            children: Box::new([
                RawLang {
                    node: "+".to_owned(),
                    children: Box::new([
                        RawLang {
                            node: "+".to_owned(),
                            children: Box::new([
                                RawLang {
                                    node: "1".to_owned(),
                                    children: Box::new([]),
                                },
                                RawLang {
                                    node: "0".to_owned(),
                                    children: Box::new([]),
                                },
                            ]),
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                    ]),
                },
                RawLang {
                    node: "+".to_owned(),
                    children: Box::new([
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: Box::new([]),
                        },
                    ]),
                },
            ]),
        };
        let rhs: RecExpr<<Halide as Trs>::Language> = "(== (+ (+ 1 0) 1) (+ 1 1))".parse().unwrap();
        assert_eq!(lhs, (&rhs).into());
    }
}
