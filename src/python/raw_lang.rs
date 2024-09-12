use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::{FromOp, Id, Language, RecExpr};
use symbolic_expressions::{Sexp, SexpError};
use thiserror::Error;

use crate::utils::Tree;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawLang {
    node: String,
    children: Vec<RawLang>,
}

impl RawLang {
    pub fn new(node: String, children: Vec<RawLang>) -> Self {
        RawLang { node, children }
    }

    pub fn name(&self) -> &str {
        &self.node
    }
}

impl Tree for RawLang {
    fn children(&self) -> &[Self] {
        self.children.as_slice()
    }

    fn children_mut(&mut self) -> &mut [Self] {
        self.children.as_mut_slice()
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
                    children: vec![],
                }),
                Sexp::List(list) if list.is_empty() => Err(RawLangParseError::EmptySexp),
                Sexp::List(list) => match &list[0] {
                    Sexp::Empty => unreachable!("Cannot be in head position"),
                    empty_list @ Sexp::List(..) => {
                        Err(RawLangParseError::HeadList(empty_list.to_owned()))
                    }
                    Sexp::String(s) => {
                        let children: Vec<RawLang> =
                            list[1..].iter().map(rec).collect::<Result<_, _>>()?;
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
    use crate::trs::halide::HalideMath;

    #[test]
    fn convert_pylang() {
        let term = "(== (+ (+ 1 0) 1) (+ 1 1))";
        let lhs: RecExpr<HalideMath> = (&term.parse::<RawLang>().unwrap()).try_into().unwrap();
        let rhs: RecExpr<HalideMath> = term.parse().unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn parse_basic() {
        let lhs = RawLang {
            node: "==".to_owned(),
            children: vec![
                RawLang {
                    node: "0".to_owned(),
                    children: vec![],
                },
                RawLang {
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
        let lhs = RawLang {
            node: "==".to_owned(),
            children: vec![
                RawLang {
                    node: "0".to_owned(),
                    children: vec![],
                },
                RawLang {
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
        let lhs = RawLang {
            node: "==".to_owned(),

            children: vec![
                RawLang {
                    node: "+".to_owned(),
                    children: vec![
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                RawLang {
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
        let lhs = RawLang {
            node: "==".to_owned(),

            children: vec![
                RawLang {
                    node: "+".to_owned(),
                    children: vec![
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                RawLang {
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
        let lhs = RawLang {
            node: "==".to_owned(),
            children: vec![
                RawLang {
                    node: "+".to_owned(),
                    children: vec![
                        RawLang {
                            node: "+".to_owned(),
                            children: vec![
                                RawLang {
                                    node: "1".to_owned(),
                                    children: vec![],
                                },
                                RawLang {
                                    node: "0".to_owned(),
                                    children: vec![],
                                },
                            ],
                        },
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                    ],
                },
                RawLang {
                    node: "+".to_owned(),
                    children: vec![
                        RawLang {
                            node: "1".to_owned(),
                            children: vec![],
                        },
                        RawLang {
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
