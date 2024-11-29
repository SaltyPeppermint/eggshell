// pub mod halide;
mod pyast;
mod raw_ast;

use std::fmt::Display;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

pub use pyast::*;

/// A wrapper around the `RecParseError` so we can circumvent the orphan rule
#[derive(Debug, Error)]
pub enum EggError<E: Display> {
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<E>),
    #[error(transparent)]
    FromOp(#[from] egg::FromOpError),
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
