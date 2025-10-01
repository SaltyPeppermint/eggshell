mod partial;
pub mod sampler;

use std::fmt::Debug;
use std::usize;

use partial::PartialRecExpr;
use thiserror::Error;

pub use sampler::Sampler;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SampleError {
    #[error("Can't convert a non-finished list of choices")]
    UnfinishedChoice,
    #[error("Extraction not possible as this eclass contains no terms of appropriate size {0}!")]
    SizeLimit(usize),
    #[error("Cannot inerst into an already filled pick!")]
    DoublePick,
}
