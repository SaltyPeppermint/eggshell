use std::fmt::{Debug, Display};

use egg::FromOp;
use pyo3::{PyErr, create_exception, exceptions::PyException};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PartialError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error("Cannot lower into an easier language: {0:?}")]
    NoLowering(String),
    #[error("No Probablility available for symbol: {0:?}")]
    NoProbability(String),
    #[error("Tried to parse zero tokens")]
    NoTokens,
    #[error("Max arity reached while trying to parse partial term: {1}: {0}")]
    MaxArity(String, usize),
    #[error("Wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error(transparent)]
    BadOp(L::Error),
}

create_exception!(
    eggshell,
    PartialException,
    PyException,
    "Error parsing a Sketch."
);

impl<L> From<PartialError<L>> for PyErr
where
    L: FromOp,
    L::Error: Display,
{
    fn from(err: PartialError<L>) -> PyErr {
        PartialException::new_err(format!("{err:?}"))
    }
}
