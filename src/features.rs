use std::fmt::Display;

use egg::Language;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    NonLeaf(usize),
    Leaf(Vec<f64>),
    IgnoredSymbol,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SymbolType<'a> {
    Operator,
    Constant(f64),
    Variable(&'a str),
    MetaSymbol,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Featurizer<L: AsFeatures> {
    symbols: Vec<L>,
    leaves: usize,
    ignore_unknown: bool,
}

impl<L: AsFeatures> Featurizer<L> {
    pub fn new(mut symbols: Vec<L>, variable_names: Vec<String>) -> Self {
        // We want the symbols with many children at the start
        symbols.extend(variable_names.into_iter().map(|name| L::into_symbol(name)));

        symbols.sort_by_key(|b| std::cmp::Reverse(b.children().len()));
        let leaves = symbols.iter().filter(|s| s.children().is_empty()).count();
        Featurizer {
            symbols,
            leaves,
            ignore_unknown: false,
        }
    }

    pub fn set_ignore_unknown(&mut self, ignore_unknown: bool) {
        self.ignore_unknown = ignore_unknown;
    }

    pub fn ignore_unknown(&self) -> bool {
        self.ignore_unknown
    }

    pub fn into_meta_lang<M: AsFeatures, F: Fn(L) -> M>(self, meta_wrapper: F) -> Featurizer<M> {
        let i = self
            .symbols
            .into_iter()
            .map(meta_wrapper)
            .collect::<Vec<M>>();
        Featurizer::new(i, vec![])
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

    fn leaves(&self) -> usize {
        self.leaves
    }

    fn non_leaves(&self) -> usize {
        self.symbols.len() - self.leaves
    }

    pub fn feature_vec_len(&self) -> usize {
        // All the leaves plus one for the constant type value
        self.leaves() + 1
    }

    pub fn features(&self, symbol: &L) -> Result<Feature, FeatureError> {
        if self.ignore_unknown && self.symbol_position(symbol).is_none() {
            return Ok(Feature::IgnoredSymbol);
        }

        if !symbol.children().is_empty() {
            return Ok(Feature::NonLeaf(
                self.symbol_position(symbol)
                    .ok_or(FeatureError::UnknownSymbol(symbol.to_string()))?,
            ));
        }
        let symbol_idx = self
            .symbol_position(symbol)
            .ok_or(FeatureError::UnknownSymbol(symbol.to_string()))?
            - self.non_leaves();

        let mut features = vec![0.0; self.feature_vec_len()];

        match symbol.symbol_type() {
            SymbolType::Operator | SymbolType::MetaSymbol | SymbolType::Variable(_) => {
                features[symbol_idx] = 1.0;
            }
            SymbolType::Constant(value) => {
                features[symbol_idx] = 1.0;
                let last_position = features.len() - 1;
                features[last_position] = value;
            }
        }
        Ok(Feature::Leaf(features))
    }

    pub fn symbols(&self) -> &[L] {
        &self.symbols
    }

    pub fn symbol_names(&self) -> Vec<String> {
        self.symbols()
            .iter()
            .map(|symbol| symbol.to_string())
            .collect()
    }
}

pub trait AsFeatures: Language + Display {
    fn featurizer(variable_names: Vec<String>) -> Featurizer<Self>;

    fn symbol_type(&self) -> SymbolType;

    fn into_symbol(name: String) -> Self;
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum FeatureError {
    #[error("Symbol not in language: {0}")]
    UnknownSymbol(String),
    #[error("Ignored Symbol has no features: {0}")]
    IgnoredSymbol(String),
    #[error("Non Leaf Node has no feature! {0}")]
    NonLeaf(String),
    #[error("Leaf has no network id: {0}")]
    Leaf(String),
}
