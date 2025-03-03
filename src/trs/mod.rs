pub mod arithmetic;
pub mod halide;
pub mod rise;
pub mod simple;

use std::fmt::{Debug, Display};

use egg::{Analysis, FromOp, Language, Rewrite};
use serde::Serialize;
use thiserror::Error;

use pyo3::{PyErr, create_exception, exceptions::PyException};

// use crate::typing::{Type, Typeable};

pub use arithmetic::Arithmetic;
pub use halide::{Halide, HalideRuleset};
pub use rise::Rise;
pub use simple::Simple;

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`TermRewriteSystem::full_rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait TermRewriteSystem {
    type Language: Language<Discriminant: Send + Sync>
        + Serialize
        + FromOp
        + Send
        + Sync
        + MetaInfo
        + 'static;
    type Analysis: Analysis<Self::Language, Data: Serialize + Clone + Send + Sync>
        + Clone
        + Serialize
        + Debug
        + Default
        + Send
        + Sync
        + 'static;

    fn full_rules() -> Vec<Rewrite<Self::Language, Self::Analysis>>;
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SymbolType<'a> {
    Operator(usize),
    Constant(usize, String),
    Variable(&'a str),
    MetaSymbol(usize),
}

pub trait MetaInfo: Display + Language {
    fn symbol_type(&self) -> SymbolType;

    #[must_use]
    fn named_symbols() -> Vec<&'static str>;

    #[must_use]
    fn n_non_vars() -> usize {
        Self::named_symbols().len() + Self::N_CONST_TYPES
    }

    const N_CONST_TYPES: usize;
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
