use egg::{Id, Language, RecExpr};

use crate::trs::{LanguageManager, MetaInfo, SymbolType, TrsError};

#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    NonLeaf(usize),
    Leaf(Vec<f64>),
    IgnoredSymbol,
}

pub fn feature_vec_len<L: MetaInfo>(lang_manager: &LanguageManager<L>) -> usize {
    // All the leaves plus one for the constant type value
    lang_manager.leaves() + 1
}

pub fn features<L: MetaInfo>(
    symbol: &L,
    lang_manager: &LanguageManager<L>,
) -> Result<Feature, TrsError> {
    if lang_manager.ignore_unknown() && lang_manager.symbol_position(symbol).is_none() {
        return Ok(Feature::IgnoredSymbol);
    }

    if !symbol.children().is_empty() {
        return Ok(Feature::NonLeaf(
            lang_manager
                .symbol_position(symbol)
                .ok_or(TrsError::UnknownSymbol(symbol.to_string()))?,
        ));
    }
    let symbol_idx = lang_manager
        .symbol_position(symbol)
        .ok_or(TrsError::UnknownSymbol(symbol.to_string()))?
        - lang_manager.non_leaves();

    let mut features = vec![0.0; feature_vec_len(lang_manager)];

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

pub trait AsFeatures<L: MetaInfo> {
    fn count_symbols(&self, lang_manager: &LanguageManager<L>) -> Result<Vec<usize>, TrsError>;

    fn arity(&self, position: usize) -> usize;

    fn size(&self) -> usize;

    fn depth(&self) -> usize;
}

impl<L: MetaInfo> AsFeatures<L> for RecExpr<L> {
    fn count_symbols(&self, lang_manager: &LanguageManager<L>) -> Result<Vec<usize>, TrsError> {
        fn rec<L: MetaInfo>(
            rec_expr: &RecExpr<L>,
            node: &L,
            lang_manager: &LanguageManager<L>,
            f: &mut Vec<usize>,
        ) -> Result<(), TrsError> {
            if let Some(p) = lang_manager.symbol_position(node) {
                f[p] += 1;
            } else if !lang_manager.ignore_unknown() {
                return Err(TrsError::UnknownSymbol(node.to_string()));
            };
            for c_id in node.children() {
                rec(rec_expr, &rec_expr[*c_id], lang_manager, f)?;
            }
            Ok(())
        }

        let symbols = lang_manager.symbols();
        let root = self.root();
        let mut f = vec![0usize; symbols.len()];
        rec(self, &self[root], lang_manager, &mut f)?;
        Ok(f)
    }
    fn arity(&self, position: usize) -> usize {
        self[Id::from(position)].children().len()
    }

    fn size(&self) -> usize {
        self.len()
    }

    fn depth(&self) -> usize {
        fn rec<IL: Language>(expr: &RecExpr<IL>, node_id: Id) -> usize {
            let node = &expr[node_id];
            1 + node
                .children()
                .iter()
                .map(|c| rec(expr, *c))
                .max()
                .unwrap_or(0)
        }
        let root = self.root();
        rec(self, root)
    }
}
