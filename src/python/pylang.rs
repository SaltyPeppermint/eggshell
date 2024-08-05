use std::fmt;
use std::fmt::{Display, Formatter};

use egg::{Language, RecExpr};
use pyo3::prelude::*;

use super::macros::pyboxable;

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyLang {
    pub(crate) x: String,
    pub(crate) xs: Vec<PyLang>,
}

pyboxable!(PyLang);

// #[pymethods]
// impl PyLang {
//     fn __eq__(&self, other: &Self) -> bool {
//         self == other
//     }
// }

impl Display for PyLang {
    #[allow(clippy::redundant_closure_for_method_calls)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.xs.is_empty() {
            write!(f, "{}", self.x)
        } else {
            let inner = self
                .xs
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            write!(f, "({} {})", self.x, inner)
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
    PyLang {
        x: node.to_string(),
        xs: node
            .children()
            .iter()
            .map(|child_id| {
                let child = &rec_expr[*child_id];
                parse_rec_expr_rec(child, rec_expr)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use crate::trs::halide::MathEquations;

    use super::*;

    #[test]
    fn parse_basic() {
        let lhs = PyLang {
            x: "==".to_string(),
            xs: vec![
                PyLang {
                    x: "0".to_string(),
                    xs: vec![],
                },
                PyLang {
                    x: "0".to_string(),
                    xs: vec![],
                },
            ],
        };
        let rhs: RecExpr<MathEquations> = "(== 0 0)".parse().unwrap();
        dbg!(&rhs);
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_basic() {
        let lhs = PyLang {
            x: "==".to_string(),
            xs: vec![
                PyLang {
                    x: "0".to_string(),
                    xs: vec![],
                },
                PyLang {
                    x: "0".to_string(),
                    xs: vec![],
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
            x: "==".to_string(),

            xs: vec![
                PyLang {
                    x: "+".to_string(),
                    xs: vec![
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                    ],
                },
                PyLang {
                    x: "2".to_string(),
                    xs: vec![],
                },
            ],
        };
        let rhs: RecExpr<MathEquations> = "(== (+ 1 1) 2)".parse().unwrap();
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_nested() {
        let lhs = PyLang {
            x: "==".to_string(),

            xs: vec![
                PyLang {
                    x: "+".to_string(),
                    xs: vec![
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                    ],
                },
                PyLang {
                    x: "2".to_string(),
                    xs: vec![],
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
            x: "==".to_string(),
            xs: vec![
                PyLang {
                    x: "+".to_string(),
                    xs: vec![
                        PyLang {
                            x: "+".to_string(),
                            xs: vec![
                                PyLang {
                                    x: "1".to_string(),
                                    xs: vec![],
                                },
                                PyLang {
                                    x: "0".to_string(),
                                    xs: vec![],
                                },
                            ],
                        },
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                    ],
                },
                PyLang {
                    x: "+".to_string(),
                    xs: vec![
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                        PyLang {
                            x: "1".to_string(),
                            xs: vec![],
                        },
                    ],
                },
            ],
        };
        let rhs: RecExpr<MathEquations> = "(== (+ (+ 1 0) 1) (+ 1 1))".parse().unwrap();
        let rhs = (&rhs).into();
        assert_eq!(lhs, rhs);
    }
}
