use std::fmt::Display;

use egg::{Language, RecExpr};
use nom::{
    branch::alt,
    character::complete::{alphanumeric1, char, multispace0},
    combinator::map,
    error::{context, VerboseError},
    multi::many0,
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};
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
        write!(f, "(")?;
        rec_fmt(self, f)?;
        write!(f, " )")
    }
}

fn rec_fmt(s_expr: &PyLang, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    match s_expr {
        PyLang::Literal { str_repr } => write!(f, " {str_repr}"),
        PyLang::Application { head, tail } => {
            rec_fmt(head, f)?;
            write!(f, " (")?;
            for inner_s_expr in tail {
                rec_fmt(inner_s_expr, f)?;
            }
            write!(f, " )")
        }
    }
}

impl<L: Language + Display> From<&RecExpr<L>> for PyLang {
    fn from(value: &RecExpr<L>) -> Self {
        let expr_str = value.to_string();
        parse_expr(&expr_str).unwrap().1
    }
}

impl From<&str> for PyLang {
    fn from(value: &str) -> Self {
        parse_expr(value).unwrap().1
    }
}

fn s_exp<'a, O, F>(inner: F) -> impl Parser<&'a str, O, VerboseError<&'a str>>
where
    F: Parser<&'a str, O, VerboseError<&'a str>>,
{
    delimited(
        context("opening paren", preceded(multispace0, char('('))),
        delimited(multispace0, inner, multispace0),
        // context("closing paren", cut(preceded(multispace0, char(')')))),
        context("closing paren", preceded(multispace0, char(')'))),
    )
}

fn parse_literal(i: &str) -> IResult<&str, PyLang, VerboseError<&str>> {
    let (i, t) = delimited(multispace0, alphanumeric1, multispace0).parse(i)?;
    println!("{i}");
    // let (i, t) = alphanumeric1(i)?;
    let literal = PyLang::Literal {
        str_repr: t.to_string(),
    };
    Ok((i, literal))
}

fn parse_head(i: &str) -> IResult<&str, Box<PyLang>, VerboseError<&str>> {
    let (i, t) = delimited(multispace0, alphanumeric1, multispace0).parse(i)?;
    let head = Box::new(PyLang::Literal {
        str_repr: t.to_string(),
    });
    Ok((i, head))
}

fn parse_tail(i: &str) -> IResult<&str, Vec<PyLang>, VerboseError<&str>> {
    let literal_list = s_exp(many0(parse_literal));
    alt((literal_list, many0(parse_expr))).parse(i)
}

fn parse_application(i: &str) -> IResult<&str, PyLang, VerboseError<&str>> {
    let f = |(head, tail)| PyLang::Application { head, tail };
    // let application_inner = map(pair(parse_head, many0(parse_expr)), f);
    let application_inner = map(pair(parse_head, parse_tail), f);

    // finally, we wrap it in an s-expression
    s_exp(application_inner).parse(i)
}

fn parse_expr(i: &str) -> IResult<&str, PyLang, VerboseError<&str>> {
    // let inner = preceded(multispace0, alt((parse_literal, parse_application)));
    // s_exp(inner).parse(i)
    preceded(multispace0, alt((parse_literal, parse_application))).parse(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "foo".to_string(),
            }),
            tail: vec![
                PyLang::Literal {
                    str_repr: "bar".to_string(),
                },
                PyLang::Literal {
                    str_repr: "baz".to_string(),
                },
                PyLang::Literal {
                    str_repr: "boo".to_string(),
                },
            ],
        };
        let rhs = "( foo ( bar baz boo ) )".into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_basic() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "foo".to_string(),
            }),
            tail: vec![
                PyLang::Literal {
                    str_repr: "bar".to_string(),
                },
                PyLang::Literal {
                    str_repr: "baz".to_string(),
                },
                PyLang::Literal {
                    str_repr: "boo".to_string(),
                },
            ],
        }
        .to_string();
        let rhs = "( foo ( bar baz boo ) )";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_nested() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "nom".to_string(),
            }),
            tail: vec![PyLang::Application {
                head: Box::new(PyLang::Literal {
                    str_repr: "nim".to_string(),
                }),
                tail: vec![
                    PyLang::Literal {
                        str_repr: "nam".to_string(),
                    },
                    PyLang::Literal {
                        str_repr: "nem".to_string(),
                    },
                ],
            }],
        };
        let rhs = "( nom ( nim ( nam nem ) ) )".into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_nested() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "nom".to_string(),
            }),
            tail: vec![PyLang::Application {
                head: Box::new(PyLang::Literal {
                    str_repr: "nim".to_string(),
                }),
                tail: vec![PyLang::Literal {
                    str_repr: "nam".to_string(),
                }],
            }],
        }
        .to_string();
        let rhs = "( nom ( nim ( nam ) ) )";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_complicated() {
        let lhs = PyLang::Application {
            head: Box::new(PyLang::Literal {
                str_repr: "a".to_string(),
            }),
            tail: vec![
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "b".to_string(),
                    }),
                    tail: vec![
                        PyLang::Literal {
                            str_repr: "c".to_string(),
                        },
                        PyLang::Literal {
                            str_repr: "d".to_string(),
                        },
                    ],
                },
                PyLang::Application {
                    head: Box::new(PyLang::Literal {
                        str_repr: "e".to_string(),
                    }),
                    tail: vec![PyLang::Literal {
                        str_repr: "f".to_string(),
                    }],
                },
            ],
        };
        let rhs: PyLang = "( a ( b ( c d ) ) ( e ( f ) ) )".into();
        assert_eq!(lhs, rhs);
    }
}
