pub mod arithmetic;
pub mod halide;
pub mod rise;
pub mod simple;

use std::fmt::{Debug, Display};

use egg::{Analysis, FromOp, Language, Rewrite};
use serde::Serialize;
use strum::{IntoEnumIterator, VariantArray};
use thiserror::Error;

use pyo3::{create_exception, exceptions::PyException, PyErr};

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
    NumericValue(f64),
    Variable(&'a str),
    MetaSymbol(usize),
}

pub trait MetaInfo: Display + Language {
    type EnumDiscriminant: Clone
        + Copy
        + Debug
        + PartialEq
        + Eq
        + Display
        + for<'a> From<&'a Self>
        + Into<&'static str>
        + VariantArray
        + IntoEnumIterator
        + 'static;

    const NON_OPERATORS: &'static [Self::EnumDiscriminant];

    fn symbol_type(&self) -> SymbolType;

    #[must_use]
    fn operator_id(&self) -> Option<usize> {
        Self::EnumDiscriminant::iter()
            .filter(|x| !Self::NON_OPERATORS.contains(x))
            .position(|x| x == self.into())
    }

    fn n_operators() -> usize {
        Self::EnumDiscriminant::VARIANTS.len() - Self::NON_OPERATORS.len()
    }

    #[must_use]
    fn operator_names() -> Vec<&'static str> {
        Self::EnumDiscriminant::iter()
            .filter(|x| !Self::NON_OPERATORS.contains(x))
            .map(|x| x.into())
            .collect()
    }
}

#[derive(Debug, Error)]
pub enum TrsError {
    #[error("Wrong number of children: {0}")]
    BadAnalysis(String),
    #[error("Bad ruleset name: {0}")]
    BadRulesetName(String),
    #[error("Symbol not in language: {0}")]
    UnknownSymbol(String),
    #[error("Ignored Symbol has no features: {0}")]
    IgnoredSymbol(String),
    #[error("Non Leaf Node has no feature! {0}")]
    UnexpectedNonLeaf(String),
    #[error("Leaf has no network id: {0}")]
    UnexpectedLeaf(String),
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
