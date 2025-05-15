// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

pub mod extract;
mod partial;
mod sketch_lang;

use std::fmt::Debug;
use std::fmt::Display;

use egg::FromOp;
use pyo3::{PyErr, create_exception, exceptions::PyException};
use thiserror::Error;

pub use partial::PartialLang;
pub use partial::PartialTerm;
pub use partial::{count_expected_tokens, lower_meta_level, partial_parse};
pub use sketch_lang::Sketch;
pub use sketch_lang::SketchLang;

#[derive(Debug, Error)]
pub enum MetaLangError<L>
where
    L: FromOp,
    L::Error: Display,
{
    #[error("Wrong number of children: {0:?}")]
    BadChildren(#[from] egg::FromOpError),
    #[error("Tried to parse a partial sketch into a full sketch")]
    PartialSketch,
    #[error("Cannot lower into an easier language: {0:?}")]
    NoLowering(String),
    #[error("No open positions in partial sketch to fill: {0:?}")]
    NoOpenPositions(egg::RecExpr<PartialLang<L>>),
    #[error("Max arity reached while trying to parse partial term: {1}: {0}")]
    MaxArity(String, usize),
    #[error(transparent)]
    BadOp(L::Error),
}

create_exception!(
    eggshell,
    SketchParseException,
    PyException,
    "Error parsing a Sketch."
);

impl<L> From<MetaLangError<L>> for PyErr
where
    L: FromOp,
    L::Error: Display,
{
    fn from(err: MetaLangError<L>) -> PyErr {
        SketchParseException::new_err(format!("{err:?}"))
    }
}
