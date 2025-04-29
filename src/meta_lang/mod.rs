// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

pub mod extract;
mod partial;
mod sketch_lang;

use std::fmt::Debug;
use std::fmt::Display;

use pyo3::{PyErr, create_exception, exceptions::PyException};
use thiserror::Error;

pub use partial::PartialLang;
pub use partial::PartialTerm;
pub use partial::lower_meta_level;
pub use partial::partial_parse;
pub use sketch_lang::Sketch;
pub use sketch_lang::SketchLang;

#[derive(Debug, Error)]
pub enum MetaLangError<E: Display> {
    #[error("Wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error("Tried to parse a partial sketch into a full sketch")]
    PartialSketch,
    #[error("Cannot lower into an easier language: {0:?}")]
    NoLowering(String),
    #[error("Max arity reached while trying to parse partial term: {1}: {0}")]
    MaxArity(String, usize),
    #[error(transparent)]
    BadOp(E),
}

create_exception!(
    eggshell,
    SketchParseException,
    PyException,
    "Error parsing a Sketch."
);

impl<E: Display> From<MetaLangError<E>> for PyErr {
    fn from(err: MetaLangError<E>) -> PyErr {
        SketchParseException::new_err(err.to_string())
    }
}
