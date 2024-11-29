use egg::Language;

pub enum SymbolType<'a> {
    Operator,
    Constant(f64),
    Variable(&'a str),
    MetaSymbol,
}

pub struct SymbolList<L: AsFeatures>(Vec<L>);

impl<L: AsFeatures> SymbolList<L> {
    pub fn new(s: Vec<L>) -> Self {
        SymbolList(s)
    }

    pub fn into_meta_lang<M: AsFeatures, F: Fn(L) -> M>(self, meta_wrapper: F) -> SymbolList<M> {
        let i = self.0.into_iter().map(meta_wrapper).collect::<Vec<M>>();
        SymbolList::new(i)
    }
}

impl<'a, L: AsFeatures> IntoIterator for &'a SymbolList<L> {
    type Item = &'a L;

    type IntoIter = std::slice::Iter<'a, L>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<L: AsFeatures> SymbolList<L> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

pub trait AsFeatures: Language {
    fn symbols() -> SymbolList<Self>;

    fn symbol_type(&self) -> SymbolType;

    fn features(&self, symbol_list: &SymbolList<Self>) -> Vec<f64> {
        match self.symbol_type() {
            SymbolType::Constant(value) => {
                let mut f = vec![0.0; symbol_list.len()];
                f.push(1.0);
                f
            }
            SymbolType::Operator => {
                // account for the const entry
                let mut f = vec![0.0; symbol_list.len() + 1];
                let p = symbol_list
                    .into_iter()
                    .position(|s| s.matches(self))
                    .expect("Must be in symbols");
                f[p] = 1.0;
                f
            }
            SymbolType::Variable(name) => todo!(),
            SymbolType::MetaSymbol => todo!(),
        }
    }

    fn empty_features(symbol_list: &SymbolList<Self>) -> Vec<f64> {
        vec![0.0; symbol_list.len()]
    }

    fn arity(&self) -> usize {
        self.children().len()
    }
}
