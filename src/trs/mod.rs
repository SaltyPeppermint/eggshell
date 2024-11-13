mod arithmetic;
mod halide;
mod rise;
mod simple;

use std::fmt::{Debug, Display};

use indexmap::IndexMap;
use thiserror::Error;

use egg::{Analysis, FromOp, Language, Rewrite};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::Serialize;

use crate::python::SymbolMetaData;
// use crate::typing::{Type, Typeable};

pub use arithmetic::Arithmetic;
pub use halide::{Halide, HalideRuleset};
pub use rise::Rise;
pub use simple::Simple;
pub(crate) use simple::SimpleLang;

pub trait TrsLang:
    Language<Discriminant: Debug + Send + Sync> + Display + Serialize + FromOp + Debug + Send + Sync
{
    fn raw_symbols() -> &'static [(&'static str, usize)];

    fn is_const(&self) -> bool;

    fn is_var(&self) -> bool;

    #[must_use]
    fn symbols(variables: usize) -> Vec<(String, usize)> {
        Self::raw_symbols()
            .iter()
            .map(|(s, a)| ((*s).to_owned(), *a))
            .chain((0..variables).map(|n| (format!("v{n}"), 0)))
            .collect()
    }

    #[must_use]
    fn symbol_lut(variables: usize) -> IndexMap<String, SymbolMetaData> {
        Self::symbols(variables)
            .into_iter()
            .map(|(name, arity)| (name, SymbolMetaData::Lang { arity }))
            .collect()
    }
}

pub trait TrsAnalysis<L: TrsLang>:
    Analysis<L, Data: Serialize + Clone + Debug + Send + Sync>
    + Clone
    + Serialize
    + Debug
    + Default
    + Send
    + Sync
{
}

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`Trs::rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait Trs: Serialize + Debug {
    type Language: TrsLang;
    type Analysis: TrsAnalysis<Self::Language>;

    fn full_rules() -> Vec<Rewrite<Self::Language, Self::Analysis>>;
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
