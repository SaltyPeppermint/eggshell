use std::fmt::Display;

use pyo3::PyErr;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use thiserror::Error;

use super::data::TreeDataError;

#[derive(Debug, Error)]
pub enum EggshellError<L: Display> {
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<L>),
    #[error(transparent)]
    TreeData(#[from] TreeDataError),
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
