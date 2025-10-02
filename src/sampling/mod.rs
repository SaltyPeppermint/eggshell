mod partial;
pub mod sampler;

use std::fmt::Debug;

use egg::StopReason;
use partial::PartialRecExpr;
use thiserror::Error;

pub use sampler::Sampler;

#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Can't convert a non-finished list of choices")]
    UnfinishedChoice,
    #[error("Extraction not possible as this eclass contains no terms of appropriate size {0}!")]
    SizeLimit(usize),
    #[error("Cannot inerst into an already filled pick!")]
    DoublePick,
    #[error("Couldn't find a suitable term after {0} retries!")]
    RetryLimit(usize),
    #[error("Could not run to appropriate iter distance!")]
    IterDistance(usize),
    #[error("Found stop reason other than IterLimit")]
    OtherStop(StopReason),
}
