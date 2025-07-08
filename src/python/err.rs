use std::fmt::Display;

use egg::FromOp;
use pyo3::PyErr;
use pyo3::exceptions::PyException;
use pyo3_stub_gen::create_exception;
use thiserror::Error;

use super::tree_data::TreeDataError;

#[derive(Debug, Error)]
pub enum EggshellError<L>
where
    L::Error: Display,
    L: Display + FromOp,
{
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<egg::FromOpError>),
    #[error(transparent)]
    BadFromOp(#[from] egg::FromOpError),
    #[error(transparent)]
    BadTreeData(#[from] TreeDataError),
    #[error(transparent)]
    PartialLang(#[from] crate::meta_lang::partial::PartialError<L>),
}

create_exception!(
    eggshell,
    EggshellException,
    PyException,
    "Eggshell internal error."
);

impl<L> From<EggshellError<L>> for PyErr
where
    L::Error: Display,
    L: Display + FromOp,
{
    fn from(err: EggshellError<L>) -> PyErr {
        EggshellException::new_err(err.to_string())
    }
}
