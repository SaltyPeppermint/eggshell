pub mod reader;
pub mod structs;

use std::io;

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IoError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Csv(#[from] csv::Error),
    #[error("Unknown Error happend!")]
    Unknown,
}

create_exception!(
    eggshell,
    IoException,
    PyException,
    "Eggshell internal error."
);

impl From<IoError> for PyErr {
    fn from(err: IoError) -> PyErr {
        IoException::new_err(err.to_string())
    }
}
