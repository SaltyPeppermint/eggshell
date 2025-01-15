mod arithmetic;
mod halide;
mod rise;
mod simple;

use std::fmt::{Debug, Display};

use hashbrown::HashMap;
use thiserror::Error;

use egg::{Analysis, FromOp, Language, Rewrite};
use pyo3::{create_exception, exceptions::PyException, PyErr};
use serde::{Deserialize, Serialize};

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
        + Sync;

    fn full_rules() -> Vec<Rewrite<Self::Language, Self::Analysis>>;
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SymbolType<'a> {
    Operator,
    Constant(f64),
    Variable(&'a str),
    MetaSymbol,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LanguageManager<L: MetaInfo> {
    symbols: Vec<L>,
    leaves: usize,
    ignore_unknown: bool,
    arity_table: HashMap<String, usize>,
}

impl<L: MetaInfo> LanguageManager<L> {
    #[must_use]
    pub fn new(mut symbols: Vec<L>, variable_names: Vec<String>) -> Self {
        // We want the symbols with many children at the start
        symbols.extend(variable_names.into_iter().map(|name| L::into_symbol(name)));

        symbols.sort_by_key(|b| std::cmp::Reverse(b.children().len()));
        let leaves = symbols.iter().filter(|s| s.is_leaf()).count();
        let arity_table = symbols
            .iter()
            .map(|s| (s.to_string(), s.children().len()))
            .collect();
        LanguageManager {
            symbols,
            leaves,
            ignore_unknown: false,
            arity_table,
        }
    }

    pub fn set_ignore_unknown(&mut self, ignore_unknown: bool) {
        self.ignore_unknown = ignore_unknown;
    }

    pub fn into_meta_lang<M: MetaInfo, F: Fn(L) -> M>(self, meta_wrapper: F) -> LanguageManager<M> {
        let i = self
            .symbols
            .into_iter()
            .map(meta_wrapper)
            .collect::<Vec<M>>();
        LanguageManager::new(i, vec![])
    }

    pub fn symbol_position(&self, symbol: &L) -> Option<usize> {
        match symbol.symbol_type() {
            SymbolType::Constant(_) => self
                .symbols
                .iter()
                .position(|s| s.discriminant() == symbol.discriminant()),
            SymbolType::Variable(name) => self.symbols.iter().position(|s| {
                if let SymbolType::Variable(other_name) = s.symbol_type() {
                    name == other_name && symbol.discriminant() == s.discriminant()
                } else {
                    false
                }
            }),
            SymbolType::Operator | SymbolType::MetaSymbol => {
                self.symbols.iter().position(|s| symbol.matches(s))
            }
        }
    }

    #[must_use]
    pub fn ignore_unknown(&self) -> bool {
        self.ignore_unknown
    }

    #[must_use]
    pub fn leaves(&self) -> usize {
        self.leaves
    }

    #[must_use]
    pub fn non_leaves(&self) -> usize {
        self.symbols.len() - self.leaves
    }

    #[must_use]
    pub fn symbols(&self) -> &[L] {
        &self.symbols
    }

    #[must_use]
    pub fn symbol_names(&self) -> Vec<String> {
        self.symbols()
            .iter()
            .map(|symbol| symbol.to_string())
            .collect()
    }
}

pub trait MetaInfo: Language + Display {
    fn manager(variable_names: Vec<String>) -> LanguageManager<Self>;

    fn symbol_type(&self) -> SymbolType;

    fn into_symbol(name: String) -> Self;
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
