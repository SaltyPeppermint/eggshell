use std::fmt::{Debug, Display};

use egg::FromOp;
use pyo3::{PyErr, create_exception, exceptions::PyException};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SketchError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error(transparent)]
    BadChildren(#[from] egg::FromOpError),
    #[error(transparent)]
    BadOp(L::Error),
}

create_exception!(
    eggshell,
    SketchException,
    PyException,
    "Error parsing a Sketch."
);

impl<L> From<SketchError<L>> for PyErr
where
    L: FromOp,
    L::Error: Display,
{
    fn from(err: SketchError<L>) -> PyErr {
        SketchException::new_err(format!("{err:?}"))
    }
}
