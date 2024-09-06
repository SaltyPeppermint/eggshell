// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

mod analysis;
pub mod extract;
mod full_sketch;
mod hashcons;
mod partial_sketch;
pub mod recursive;
mod utils;

use std::fmt::Display;

use pyo3::{create_exception, exceptions::PyException, PyErr};
use thiserror::Error;

pub use full_sketch::Sketch;
pub use full_sketch::SketchNode;
pub use partial_sketch::PartialSketch;
pub use partial_sketch::PartialSketchNode;

#[derive(Debug, Error)]
pub enum SketchParseError<E: Display> {
    #[error("Wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error("Tried to parse a partial sketch into a full sketch")]
    PartialSketch,
    #[error(transparent)]
    BadOp(E),
}

create_exception!(
    eggshell,
    SketchParseException,
    PyException,
    "Error parsing a Sketch."
);

impl<E: Display> From<SketchParseError<E>> for PyErr {
    fn from(err: SketchParseError<E>) -> PyErr {
        SketchParseException::new_err(err.to_string())
    }
}
