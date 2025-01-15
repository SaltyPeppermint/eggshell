use std::fmt::Display;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EggshellError<L: Display> {
    // #[error(transparent)]
    // Io(#[from] std::io::Error),
    // #[error(transparent)]
    // Csv(#[from] csv::Error),
    #[error(transparent)]
    Trs(#[from] crate::trs::TrsError),
    #[error(transparent)]
    Sample(#[from] crate::sampling::SampleError),
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<L>),
    #[error(transparent)]
    FromOp(#[from] egg::FromOpError),
    #[error("Unknown Error happend!")]
    Unknown,
}

create_exception!(
    eggshell,
    EggshellException,
    PyException,
    "Eggshell internal error."
);

impl<L: Display> From<EggshellError<L>> for PyErr {
    fn from(err: EggshellError<L>) -> PyErr {
        EggshellException::new_err(err.to_string())
    }
}
