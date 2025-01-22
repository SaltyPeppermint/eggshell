use egg::{Id, Language, RecExpr};

use crate::trs::{MetaInfo, SymbolType, TrsError};

pub fn features<L: MetaInfo, S: AsRef<str>>(
    symbol: &L,
    variable_names: &[S],
    ignore_unknown: bool,
) -> Result<Option<Vec<f64>>, TrsError> {
    if ignore_unknown && matches!(symbol.symbol_type(), SymbolType::Variable(_)) {
        return Ok(None);
    }

    let mut features = vec![0.0; feature_vec_len::<L, S>(variable_names)];

    match symbol.symbol_type() {
        SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => {
            features[idx] = 1.0;
        }
        SymbolType::Constant(value) => {
            let constant_idx = L::operators().len() + 1;
            let const_value_idx = L::operators().len() + 2;
            features[constant_idx] = 1.0;
            features[const_value_idx] = value;
        }

        SymbolType::Variable(name) => {
            if let Some(variable_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                features[L::operators().len() + 3 + variable_idx] = 1.0;
            } else {
                return Err(TrsError::UnknownSymbol(name.to_owned()));
            }
        }
    }
    Ok(Some(features))
}

fn feature_vec_len<L: MetaInfo, S>(variable_names: &[S]) -> usize {
    // All the leaves
    // plus two for the constant type and its value
    // plus n for the variable names
    L::operators().len() + 2 + variable_names.len()
}

pub trait AsFeatures<L: MetaInfo> {
    fn count_symbols<S: AsRef<str>>(
        &self,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<Vec<usize>, TrsError>;
    fn arity(&self, position: usize) -> usize;

    fn size(&self) -> usize;

    fn depth(&self) -> usize;
}

impl<L: MetaInfo> AsFeatures<L> for RecExpr<L> {
    fn count_symbols<S: AsRef<str>>(
        &self,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<Vec<usize>, TrsError> {
        fn rec<L: MetaInfo, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            ignore_unknown: bool,
            f: &mut Vec<usize>,
        ) -> Result<(), TrsError> {
            match node.symbol_type() {
                SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => f[idx] += 1,
                SymbolType::Constant(_) => f[L::operators().len() + 1] += 1,
                SymbolType::Variable(name) => {
                    if let Some(var_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                        f[L::operators().len() + 2 + var_idx] += 1;
                    } else if !ignore_unknown {
                        return Err(TrsError::UnknownSymbol(name.to_owned()));
                    }
                }
            }
            for c_id in node.children() {
                rec(
                    rec_expr,
                    &rec_expr[*c_id],
                    variable_names,
                    ignore_unknown,
                    f,
                )?;
            }
            Ok(())
        }

        let root = self.root();
        // All operators, one for const, and variable_names
        let mut f = vec![0usize; L::operators().len() + 1 + variable_names.len()];
        rec(self, &self[root], variable_names, ignore_unknown, &mut f)?;
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
        rec(self, self.root())
    }
}
