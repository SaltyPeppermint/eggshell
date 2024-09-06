pub mod base;
pub mod sketch;

use pyo3::{create_exception, exceptions::PyException, PyErr};

use thiserror::Error;

pub use base::Typeable;

#[derive(Debug, Error)]
pub enum TypingError {
    #[error("Incomparable types {0} {1}")]
    Incomparable(String, String),
    #[error("Type constraint {constraint} incomparable with childrens type : {found}")]
    ConstraintViolation { constraint: String, found: String },
}

create_exception!(
    eggshell,
    TypingException,
    PyException,
    "Eggshell internal error."
);

impl From<TypingError> for PyErr {
    fn from(err: TypingError) -> PyErr {
        TypingException::new_err(err.to_string())
    }
}
