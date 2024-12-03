use egg::Language;

#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    NonLeaf(usize),
    Leaf(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SymbolType<'a> {
    Operator,
    Constant(f64),
    Variable(&'a str),
    MetaSymbol,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Featurizer<L: AsFeatures> {
    symbols: Vec<L>,
    leaves: usize,
}

impl<L: AsFeatures> Featurizer<L> {
    pub fn new(mut symbols: Vec<L>, variable_names: Vec<String>) -> Self {
        // We want the symbols with many children at the start
        symbols.extend(variable_names.into_iter().map(|name| L::into_symbol(name)));

        symbols.sort_by_key(|b| std::cmp::Reverse(b.children().len()));
        let leaves = symbols.iter().filter(|s| s.children().is_empty()).count();
        Featurizer { symbols, leaves }
    }

    pub fn into_meta_lang<M: AsFeatures, F: Fn(L) -> M>(self, meta_wrapper: F) -> Featurizer<M> {
        let i = self
            .symbols
            .into_iter()
            .map(meta_wrapper)
            .collect::<Vec<M>>();
        Featurizer::new(i, vec![])
    }

    fn symbol_position(&self, symbol: &L) -> Option<usize> {
        if let SymbolType::Constant(_) = symbol.symbol_type() {
            self.symbols
                .iter()
                .position(|s| s.discriminant() == symbol.discriminant())
        } else {
            self.symbols.iter().position(|s| symbol.matches(s))
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

    pub fn features(&self, symbol: &L) -> Feature {
        if !symbol.children().is_empty() {
            return Feature::NonLeaf(self.symbol_position(symbol).unwrap());
        }

        let symbol_idx = self
            .symbol_position(symbol)
            .expect("Do not call on symbols with children")
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
        Feature::Leaf(features)
    }

    pub fn symbols(&self) -> &[L] {
        &self.symbols
    }
}

pub trait AsFeatures: Language {
    fn symbol_list(variable_names: Vec<String>) -> Featurizer<Self>;

    fn symbol_type(&self) -> SymbolType;

    fn into_symbol(name: String) -> Self;
}
