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
    #[error("IO Error during file loading")]
    IO(#[from] std::io::Error),
    #[error("Error during CSV parsing")]
    Csv(#[from] std::num::ParseFloatError),
    #[error("Error during parsing a float")]
    FloatParse(#[from] csv::Error),
    #[error("No Equality Saturation was run on the Egraph!")]
    MissingEqsat,
    #[error("Already returned a result!")]
    AlreadyFinished,
    #[error("Could not parse the given term: {0}")]
    TermParse(String),
    #[error("You tried to unpack the wrong tuple! This is a: {0}")]
    TupleUnpacking(String),
    #[error("Unknown Error happend!")]
    Unknown,
}

impl std::convert::From<EggShellError> for PyErr {
    fn from(err: EggShellError) -> PyErr {
        EggShellException::new_err(err.to_string())
    }
}
