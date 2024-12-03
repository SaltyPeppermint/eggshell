// pub mod halide;
mod pyast;
mod raw_ast;

use std::fmt::Display;

use thiserror::Error;

pub use pyast::*;

/// A wrapper around the `RecParseError` so we can circumvent the orphan rule
#[derive(Debug, Error)]
pub enum EggError<E: Display> {
    #[error(transparent)]
    RecExprParse(#[from] egg::RecExprParseError<E>),
    #[error(transparent)]
    FromOp(#[from] egg::FromOpError),
}
