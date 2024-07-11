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
    #[error("No Equality Saturation was run on the Egraph!")]
    MissingEqsat,
    #[error("Already returned a result!")]
    AlreadyFinished,
    #[error("Could not parse the given term: {0}")]
    TermParse(String),
    #[error("Error occured during parsing/reading")]
    Io(String),
    #[error("You tried to unpack the wrong tuple! This is a: {0}")]
    TupleUnpacking(String),
    #[error("Unknown Error happend!")]
    Unknown,
}

impl From<csv::Error> for EggShellError {
    fn from(value: csv::Error) -> Self {
        EggShellError::Io(value.to_string())
    }
}

impl From<io::Error> for EggShellError {
    fn from(value: io::Error) -> Self {
        EggShellError::Io(value.to_string())
    }
}

impl From<ParseFloatError> for EggShellError {
    fn from(value: ParseFloatError) -> Self {
        EggShellError::Io(value.to_string())
    }
}

impl std::convert::From<EggShellError> for PyErr {
    fn from(err: EggShellError) -> PyErr {
        EggShellException::new_err(err.to_string())
    }
}
