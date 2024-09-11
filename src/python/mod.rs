// pub mod halide;
mod flat;
pub(crate) mod macros;
mod pylang;
mod pysketch;
mod pytrs;
mod raw_lang;
mod raw_sketch;

use std::fmt::Display;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

pub use flat::FlatAst;
pub use flat::FlatNode;
pub use pylang::PyLang;
pub use pysketch::PySketch;
pub use pytrs::*;

pub(crate) use raw_lang::RawLang;
pub(crate) use raw_sketch::RawSketch;

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
