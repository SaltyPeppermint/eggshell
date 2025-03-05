pub mod arithmetic;
pub mod halide;
pub mod rise;
pub mod simple;

use std::fmt::{Debug, Display};

use egg::{Analysis, FromOp, Language, Rewrite};
use serde::Serialize;
use strum::EnumCount;
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

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub struct SymbolInfo {
    id: usize,
    symbol_type: SymbolType,
}

impl SymbolInfo {
    #[must_use]
    pub fn new(id: usize, symbol_type: SymbolType) -> Self {
        Self { id, symbol_type }
    }

    #[must_use]
    pub fn value(&self) -> Option<String> {
        match &self.symbol_type {
            SymbolType::Constant(v) | SymbolType::Variable(v) => Some(v.to_owned()),
            SymbolType::MetaSymbol | SymbolType::Operator => None,
        }
    }

    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub enum SymbolType {
    Operator,
    Constant(String),
    Variable(String),
    MetaSymbol,
}

pub trait MetaInfo: Display + Language + EnumCount {
    fn symbol_info(&self) -> SymbolInfo;

    #[must_use]
    fn operators() -> Vec<&'static str>;

    const NUM_SYMBOLS: usize = Self::COUNT;
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
