// pub mod halide;
mod flat;
mod macros;
mod pylang;
mod pysketch;
mod pytrs;
mod symbols;

use std::fmt::Display;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

pub use flat::{FlatAst, FlatEGraph, FlatNode, FlatVertex};
pub use pylang::PyLang;
pub use pysketch::PySketch;
pub use pytrs::*;

pub(crate) use pylang::RawLang;
pub(crate) use pysketch::RawSketch;
pub(crate) use symbols::{SymbolMetaData, SymbolTable};

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
