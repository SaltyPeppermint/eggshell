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
    #[error("Error occured during IO: {0}")]
    Io(#[from] io::Error),
    #[error("Error occured during Float Parsing: {0}")]
    FloatParse(#[from] ParseFloatError),
    #[error("Error occured during CSV parsing: {0}")]
    Csv(#[from] csv::Error),
    #[error("Error occured during JSON Serialization: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("Invalid Argument: {0}")]
    InvalidArgument(String),
    #[error("Unknown Error happend!")]
    Unknown,
}

impl std::convert::From<EggShellError> for PyErr {
    fn from(err: EggShellError) -> PyErr {
        EggShellException::new_err(err.to_string())
    }
}
