pub mod arithmetic;
pub mod halide;
pub mod rise;
pub mod simple;

use std::fmt::{Debug, Display};

use egg::{Analysis, FromOp, Language, Rewrite};
use serde::Serialize;
use thiserror::Error;

pub use arithmetic::Arithmetic;
pub use halide::{Halide, HalideRuleset};
pub use rise::Rise;
pub use simple::Simple;

/// Trait that must be implemented by all RewriteSystem consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`RewriteSystem::full_rules`] returns the vector of [`Rewrite`] of your [`RewriteSystem`], specified
/// by your ruleset class.
pub trait RewriteSystem {
    type Language: Language<Discriminant: Send + Sync>
        + Serialize
        + FromOp
        + Display
        + Send
        + Sync
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

    #[must_use]
    pub fn symbol_type(&self) -> &SymbolType {
        &self.symbol_type
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize)]
pub enum SymbolType {
    Operator,
    Constant(String),
    Variable(String),
    MetaSymbol,
}

#[derive(Debug, Error)]
pub enum RewriteSystemError {
    #[error("Wrong number of children: {0}")]
    BadAnalysis(String),
    #[error("Bad ruleset name: {0}")]
    BadRulesetName(String),
}
