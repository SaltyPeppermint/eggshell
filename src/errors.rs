use std::io;
use std::num::ParseFloatError;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

create_exception!(
    eggshell,
    EggShellException,
    PyException,
    "Eggshell internal error."
);

#[derive(Error, Debug)]
pub enum EggShellError {
    #[error("Could not parse the given term: {0}")]
    TermParse(String),
    #[error("Error occured during IO")]
    Io(#[from] io::Error),
    #[error("Error occured during float parsing")]
    FloatParse(#[from] ParseFloatError),
    #[error("Error occured during csv parsing")]
    Csv(#[from] csv::Error),
    #[error("Error occured during JSON serialization")]
    Serialize(#[from] serde_json::Error),
    #[error("Invalid Argument")]
    InvalidArgument(String),
    #[error("Unknown Error happend!")]
    Unknown,
}

impl std::convert::From<EggShellError> for PyErr {
    fn from(err: EggShellError) -> PyErr {
        EggShellException::new_err(err.to_string())
    }
}
