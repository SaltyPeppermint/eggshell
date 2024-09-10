pub mod arithmetic;
pub mod halide;
pub mod simple;

pub use arithmetic::Arithmetic;
pub use halide::Halide;
pub use simple::Simple;
use thiserror::Error;

use std::fmt::Display;

use egg::{Analysis, FromOp, Language, Rewrite};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::Serialize;

use crate::typing::{Type, Typeable};

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`Trs::rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait Trs: Serialize {
    type Language: Display + Serialize + FromOp + Typeable<Type: Type> + SymbolIter;
    type Analysis: Analysis<Self::Language, Data: Serialize + Clone> + Clone + Serialize + Default;
    type Rulesets: TryFrom<String>;

    fn rules(ruleset: &Self::Rulesets) -> Vec<Rewrite<Self::Language, Self::Analysis>>;
}

pub trait SymbolIter: Language {
    fn raw_symbols() -> &'static [(&'static str, usize)];

    #[must_use]
    fn symbols(variables: usize, constants: usize) -> impl Iterator<Item = (String, usize)> {
        Self::raw_symbols()
            .iter()
            .map(|(s, a)| ((*s).to_owned(), *a))
            .chain((0..variables).map(|n| (format!("v{n}"), 0)))
            .chain((0..constants).map(|n| (n.to_string(), 0)))
    }
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
