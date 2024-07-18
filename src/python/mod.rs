// pub mod halide;
mod macros;
mod pytrs;

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

pub use pytrs::*;

#[pyclass(frozen)]
#[derive(Debug, Clone)]
pub enum PyLang {
    Literal {
        str_repr: String,
    },
    /// (func-name arg1 arg2)
    Application {
        /// Can't use a Box<PyLang> cause then pyo3 would freak out
        head: Py<PyLang>,
        tail: Vec<PyLang>,
    },
}

impl PyLang {
    fn rec_fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            PyLang::Literal { str_repr } => write!(f, " {str_repr}"),
            PyLang::Application { head, tail } => {
                head.get().rec_fmt(f)?;
                write!(f, " (")?;
                for t in tail {
                    t.rec_fmt(f)?;
                }
                write!(f, " )")
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl PartialEq for PyLang {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Literal { str_repr: l_str }, Self::Literal { str_repr: r_str }) => {
                l_str == r_str
            }
            (
                Self::Application {
                    head: l_head,
                    tail: l_tail,
                },
                Self::Application {
                    head: r_head,
                    tail: r_tail,
                },
            ) => l_head.get() == r_head.get() && l_tail == r_tail,
            _ => false,
        }
    }
}

impl Display for PyLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        self.rec_fmt(f)?;
        write!(f, " )")
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

fn parse_head(i: &str) -> IResult<&str, PyLang, VerboseError<&str>> {
    let (i, t) = delimited(multispace0, alphanumeric1, multispace0).parse(i)?;
    let head = PyLang::Literal {
        str_repr: t.to_string(),
    };
    Ok((i, head))
}

fn parse_tail(i: &str) -> IResult<&str, Vec<PyLang>, VerboseError<&str>> {
    let literal_list = s_exp(many0(parse_literal));
    alt((literal_list, many0(parse_expr))).parse(i)
}

fn parse_application(i: &str) -> IResult<&str, PyLang, VerboseError<&str>> {
    let f = |(head, tail)| PyLang::Application {
        head: Python::with_gil(|py| Py::new(py, head)).unwrap(),
        tail,
    };
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
    fn parse_basic_str() {
        let lhs = Python::with_gil(|py| PyLang::Application {
            head: Py::new(
                py,
                PyLang::Literal {
                    str_repr: "foo".to_string(),
                },
            )
            .unwrap(),
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
        });
        let rhs = "( foo ( bar baz boo ) )".into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_basic_str() {
        let lhs = Python::with_gil(|py| PyLang::Application {
            head: Py::new(
                py,
                PyLang::Literal {
                    str_repr: "foo".to_string(),
                },
            )
            .unwrap(),
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
        })
        .to_string();
        let rhs = "( foo ( bar baz boo ) )";
        assert_eq!(&lhs, rhs);
    }

    #[test]
    fn parse_nested_str() {
        let lhs = Python::with_gil(|py| PyLang::Application {
            head: Py::new(
                py,
                PyLang::Literal {
                    str_repr: "nom".to_string(),
                },
            )
            .unwrap(),
            tail: vec![PyLang::Application {
                head: Py::new(
                    py,
                    PyLang::Literal {
                        str_repr: "nim".to_string(),
                    },
                )
                .unwrap(),
                tail: vec![
                    PyLang::Literal {
                        str_repr: "nam".to_string(),
                    },
                    PyLang::Literal {
                        str_repr: "nem".to_string(),
                    },
                ],
            }],
        });
        let rhs = "( nom ( nim ( nam nem ) ) )".into();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn print_nested_str() {
        let lhs = Python::with_gil(|py| PyLang::Application {
            head: Py::new(
                py,
                PyLang::Literal {
                    str_repr: "nom".to_string(),
                },
            )
            .unwrap(),
            tail: vec![PyLang::Application {
                head: Py::new(
                    py,
                    PyLang::Literal {
                        str_repr: "nim".to_string(),
                    },
                )
                .unwrap(),
                tail: vec![PyLang::Literal {
                    str_repr: "nam".to_string(),
                }],
            }],
        })
        .to_string();
        let rhs = "( nom ( nim ( nam ) ) )";
        assert_eq!(&lhs, rhs);
    }
}
