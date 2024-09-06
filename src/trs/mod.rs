pub mod arithmetic;
pub mod halide;
pub mod simple;

pub use arithmetic::Arithmetic;
pub use halide::Halide;
pub use simple::Simple;
use thiserror::Error;

use std::fmt::Display;

use egg::{Analysis, FromOp, Rewrite};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::Serialize;

use crate::typing::Typeable;

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`Trs::rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait Trs: Serialize {
    type Language: Display + Serialize + FromOp + Typeable<Type: PartialOrd + Eq>;
    type Analysis: Analysis<Self::Language, Data: Serialize + Clone> + Clone + Serialize + Default;
    type Rulesets: TryFrom<String>;

    fn rules(ruleset_class: &Self::Rulesets) -> Vec<Rewrite<Self::Language, Self::Analysis>>;
}

#[derive(Debug, Error)]
pub enum TrsError {
    #[error("Wrong number of children: {0}")]
    BadAnalysis(String),
    #[error("Bad ruleset name: {0}")]
    BadRulesetName(String),
}

create_exception!(
    eggshell,
    TrsException,
    PyException,
    "Eggshell internal error."
);

impl From<TrsError> for PyErr {
    fn from(err: TrsError) -> PyErr {
        TrsException::new_err(err.to_string())
    }
}
