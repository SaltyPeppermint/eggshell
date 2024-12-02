use egg::Language;

#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    NonLeaf(usize),
    Leaf(Vec<f64>),
}

#[derive(PartialEq, PartialOrd)]
pub enum SymbolType<'a> {
    Operator,
    Constant(f64),
    Variable(&'a str),
    MetaSymbol,
}

pub struct Featurizer<L: AsFeatures> {
    symbols: Vec<L>,
    variable_names: Vec<String>,
    leaves: usize,
}

impl<L: AsFeatures> Featurizer<L> {
    pub fn new(mut symbols: Vec<L>, variable_names: Vec<String>) -> Self {
        // We want the symbols with many children at the start
        symbols.sort_by_key(|b| std::cmp::Reverse(b.children().len()));
        let leaves =
            symbols.iter().filter(|s| s.children().is_empty()).count() + variable_names.len();
        Featurizer {
            symbols,
            variable_names,
            leaves,
        }
    }

    pub fn into_meta_lang<M: AsFeatures, F: Fn(L) -> M>(self, meta_wrapper: F) -> Featurizer<M> {
        let i = self
            .symbols
            .into_iter()
            .map(meta_wrapper)
            .collect::<Vec<M>>();
        Featurizer::new(i, self.variable_names)
    }

    fn symbol_position(&self, symbol: &L) -> Option<usize> {
        self.symbols.iter().position(|s| symbol.matches(s))
    }

    fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    fn leaves(&self) -> usize {
        self.leaves
    }

    pub fn features(&self, symbol: &L) -> Feature {
        if !symbol.children().is_empty() {
            return Feature::NonLeaf(self.symbol_position(symbol).unwrap());
        }

        let symbol_idx = self
            .symbol_position(symbol)
            .expect("DO NOT CALL ON SYMBOLS WITH CHILDREN")
            - self.leaves();

        let mut features = self.empty_features();

        match symbol.symbol_type() {
            SymbolType::Operator | SymbolType::MetaSymbol => {
                features[symbol_idx] = 1.0;
            }
            SymbolType::Constant(value) => {
                features[symbol_idx] = 1.0;
                let last_position = features.len() - 1;
                features[last_position] = value;
            }
            SymbolType::Variable(name) => {
                features[symbol_idx] = 1.0;

                let var_idx = self
                    .variable_names()
                    .iter()
                    .position(|s| s.as_str() == name)
                    .unwrap();
                features[self.leaves() + var_idx] = 1.0;
            }
        }
        Feature::Leaf(features)
    }

    fn empty_features(&self) -> Vec<f64> {
        vec![0.0; self.leaves() + self.variable_names().len() + 1]
    }
}

pub trait AsFeatures: Language {
    fn symbol_list(variable_names: Vec<String>) -> Featurizer<Self>;

    fn symbol_type(&self) -> SymbolType;
}
