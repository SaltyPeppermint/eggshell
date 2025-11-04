pub mod arithmetic;
pub mod dummy_rise;
pub mod halide;
pub mod herbie;
pub mod rise;
pub mod simple;

use std::fmt::Debug;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RewriteSystemError {
    #[error("Wrong number of children: {0}")]
    BadAnalysis(String),
    #[error("Bad ruleset name: {0}")]
    BadRulesetName(String),
}
