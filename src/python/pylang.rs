use std::fmt::Display;

use egg::{Language, RecExpr};
use pyo3::prelude::*;

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyLang {
    Literal {
        str_repr: String,
    },
    /// (func-name arg1 arg2)
    Application {
        head: Box<PyLang>,
        tail: Vec<PyLang>,
    },
}

super::macros::pyboxable!(PyLang);

#[pymethods]
impl PyLang {
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl Display for PyLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyLang::Literal { str_repr } => write!(f, "{str_repr}"),
            PyLang::Application { head, tail } => {
                write!(f, "( ")?;
                head.fmt(f)?;
                write!(f, " ")?;
                for inner_s_expr in tail {
                    inner_s_expr.fmt(f)?;
                    write!(f, " ")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<L: Language + Display> From<&RecExpr<L>> for PyLang {
    fn from(rec_expr: &RecExpr<L>) -> Self {
        // See https://docs.rs/egg/latest/egg/struct.RecExpr.html
        // "RecExprs must satisfy the invariant that enodes’ children must refer to elements that come before it in the list."
        // Therefore, in a RecExpr that has only one root, the last element must be the root.
        let root = rec_expr.as_ref().last().unwrap();
        parse_rec_expr_rec(root, rec_expr)
    }
}

fn parse_rec_expr_rec<L: Language + Display>(node: &L, rec_expr: &RecExpr<L>) -> PyLang {
    if node.is_leaf() {
        PyLang::Literal {
            str_repr: node.to_string(),
        }
    } else {
        PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: node.to_string(),
            }),
            tail: node
                .children()
                .iter()
                .map(|child_id| {
                    let child = &rec_expr[*child_id];
                    parse_rec_expr_rec(child, rec_expr)
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trs::halide::Math;

    use super::*;

    #[test]
    fn parse_basic() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "==".to_string(),
            }),
            tail: vec![
                PyLang::Literal {
                    str_repr: "0".to_string(),
                },
                PyLang::Literal {
                    str_repr: "0".to_string(),
                },
            ],
        };
        let rhs: RecExpr<Math> = "( == 0 0 )".parse().unwrap();
        dbg!(&rhs);
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_basic() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "==".to_string(),
            }),
            tail: vec![
                PyLang::Literal {
                    str_repr: "0".to_string(),
                },
                PyLang::Literal {
                    str_repr: "0".to_string(),
                },
            ],
        }
        .to_string();
        let rhs = "( == 0 0 )";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_nested() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "==".to_string(),
            }),
            tail: vec![
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "+".to_string(),
                    }),
                    tail: vec![
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                    ],
                },
                PyLang::Literal {
                    str_repr: "2".to_string(),
                },
            ],
        };
        let rhs: RecExpr<Math> = "( == ( + 1 1 ) 2 )".parse().unwrap();
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_nested() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "==".to_string(),
            }),
            tail: vec![
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "+".to_string(),
                    }),
                    tail: vec![
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                    ],
                },
                PyLang::Literal {
                    str_repr: "2".to_string(),
                },
            ],
        }
        .to_string();
        let rhs = "( == ( + 1 1 ) 2 )";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_complex() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "==".to_string(),
            }),
            tail: vec![
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "+".to_string(),
                    }),
                    tail: vec![
                        PyLang::Application {
                            head: Box::new(PyLang::Literal {
                                str_repr: "+".to_string(),
                            }),
                            tail: vec![
                                PyLang::Literal {
                                    str_repr: "1".to_string(),
                                },
                                PyLang::Literal {
                                    str_repr: "0".to_string(),
                                },
                            ],
                        },
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                    ],
                },
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "+".to_string(),
                    }),
                    tail: vec![
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                        PyLang::Literal {
                            str_repr: "1".to_string(),
                        },
                    ],
                },
            ],
        };
        let rhs: RecExpr<Math> = "( == ( + ( + 1 0 ) 1 ) ( + 1 1 ) )".parse().unwrap();
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }
}
