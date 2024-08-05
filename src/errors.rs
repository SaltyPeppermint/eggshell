use std::fmt::Display;
use std::io;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EggShellError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error("Unknown Error happend!")]
    Unknown,
}

create_exception!(
    eggshell,
    EggShellException,
    PyException,
    "Eggshell internal error."
);

impl From<EggShellError> for PyErr {
    fn from(err: EggShellError) -> PyErr {
        EggShellException::new_err(err.to_string())
    }
}

#[derive(Debug, Error)]
pub enum EggError<E: Display> {
    #[error(transparent)]
    FromOp(#[from] egg::FromOpError),
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<E>),
}

create_exception!(
    eggshell,
    EggException,
    PyException,
    "Eggshell internal error."
);

impl<E: Display> From<EggError<E>> for PyErr {
    fn from(err: EggError<E>) -> PyErr {
        EggException::new_err(err.to_string())
    }
}

#[derive(Debug, Error)]
pub enum SketchParseError<E: Display> {
    #[error("wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error(transparent)]
    BadOp(E),
}

create_exception!(
    eggshell,
    SketchParseException,
    PyException,
    "Eggshell internal error."
);

impl<E: Display> From<SketchParseError<E>> for PyErr {
    fn from(err: SketchParseError<E>) -> PyErr {
        SketchParseException::new_err(err.to_string())
    }
}
