use egg::{FromOp, Id, Language, RecExpr};
use serde::Serialize;

use crate::trs::{MetaInfo, SymbolType, TrsError};

pub fn features<L: MetaInfo, S: AsRef<str>>(
    symbol: &L,
    variable_names: &[S],
    ignore_unknown: bool,
) -> Result<Option<Vec<f64>>, TrsError> {
    if ignore_unknown && matches!(symbol.symbol_type(), SymbolType::Variable(_)) {
        return Ok(None);
    }

    // All the leaves
    // plus two for the constant type and its value
    // plus n for the variable names
    let mut features = vec![0.0; L::operator_names().len() + 2 + variable_names.len()];

    match symbol.symbol_type() {
        SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => {
            features[idx] = 1.0;
        }
        SymbolType::NumericValue(value) => {
            let constant_idx = L::operator_names().len() + 1;
            let const_value_idx = L::operator_names().len() + 2;
            features[constant_idx] = 1.0;
            features[const_value_idx] = value;
        }

        SymbolType::Variable(name) => {
            if let Some(variable_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                features[L::operator_names().len() + 3 + variable_idx] = 1.0;
            } else {
                return Err(TrsError::UnknownSymbol(name.to_owned()));
            }
        }
    }
    Ok(Some(features))
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

impl<L: Language + MetaInfo> AsFeatures<L> for RecExpr<L> {
    fn count_symbols<S: AsRef<str>>(
        &self,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<Vec<usize>, TrsError> {
        fn rec<L: Language + MetaInfo, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            ignore_unknown: bool,
            f: &mut Vec<usize>,
        ) -> Result<(), TrsError> {
            match node.symbol_type() {
                SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => f[idx] += 1,
                // right behind the operators len for the constant type
                SymbolType::NumericValue(_) => f[L::operator_names().len()] += 1,
                SymbolType::Variable(name) => {
                    if let Some(var_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                        // 1 since we count as all the same
                        f[L::operator_names().len() + var_idx] += 1;
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
        let mut f = vec![0usize; L::operator_names().len() + 1 + variable_names.len()];
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

#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct GraphData {
    nodes: Vec<Vec<f64>>,
    edges: [Vec<usize>; 2],
}

impl GraphData {
    pub fn nodes(&self) -> &[Vec<f64>] {
        &self.nodes
    }

    pub fn edges(&self) -> &[Vec<usize>; 2] {
        &self.edges
    }

    pub fn new<S: AsRef<str>, L: MetaInfo>(
        rec_expr: &RecExpr<L>,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<GraphData, TrsError> {
        fn rec<L: Language + MetaInfo, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            ignore_unknown: bool,
            nodes: &mut Vec<Vec<f64>>,
            edges: &mut [Vec<usize>; 2],
        ) -> Result<(), TrsError> {
            // All operators, variable_names, and last two are the constant symbol with its value
            let mut node_feature = initialize_feature_vector::<L, S>(variable_names);
            match node.symbol_type() {
                SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => node_feature[idx] = 1.0,
                // right behind the operators len for the constant type
                SymbolType::NumericValue(v) => {
                    let node_feature_len = node_feature.len();
                    node_feature[node_feature_len - 2] = 1.0;
                    node_feature[node_feature_len - 1] = v;
                }
                SymbolType::Variable(name) => {
                    if let Some(var_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                        // 1 since we count as all the same
                        node_feature[L::operator_names().len() + var_idx] = 1.0;
                    } else if !ignore_unknown {
                        return Err(TrsError::UnknownSymbol(name.to_owned()));
                    }
                }
            };
            nodes.push(node_feature);
            let parent_position = nodes.len() - 1;
            for c_id in node.children() {
                let child_position = nodes.len();
                edges[0].push(parent_position);
                edges[1].push(child_position);
                rec(
                    rec_expr,
                    &rec_expr[*c_id],
                    variable_names,
                    ignore_unknown,
                    nodes,
                    edges,
                )?;
            }
            Ok(())
        }

        let root = rec_expr.root();
        // All operators, one for const, and variable_names
        let mut nodes = Vec::new();
        let mut edges = [Vec::new(), Vec::new()];
        rec(
            rec_expr,
            &rec_expr[root],
            variable_names,
            ignore_unknown,
            &mut nodes,
            &mut edges,
        )?;
        Ok(GraphData { nodes, edges })
    }

    pub fn to_rec_expr<L: MetaInfo + FromOp, S: AsRef<str>>(
        &self,
        variable_names: &[S],
    ) -> RecExpr<L> {
        fn rec<LL: MetaInfo + FromOp, SS: AsRef<str>>(
            data: &GraphData,
            variable_names: &[SS],
            node_idx: usize,
            stack: &mut Vec<LL>,
        ) -> Id {
            let children = data.edges()[0]
                .iter()
                .zip(data.edges()[1].iter())
                .filter(|(p, _)| **p == node_idx)
                .map(|(_, c)| *c)
                .map(|child_idx| rec::<LL, SS>(data, variable_names, child_idx, stack))
                .collect();
            let node = data.feature_vec_to_node::<LL, SS>(node_idx, children, variable_names);
            stack.push(node);
            Id::from(stack.len() - 1)
        }

        // Only one node has nothing pointing to it: The root
        let root = self.edges[0]
            .iter()
            .find(|x| !self.edges[1].contains(x))
            .unwrap();

        let mut stack = vec![];
        rec(self, variable_names, *root, &mut stack);
        stack.into()
    }

    fn feature_vec_to_node<L: Language + MetaInfo + FromOp, S: AsRef<str>>(
        &self,
        node_idx: usize,
        children: Vec<Id>,
        variable_names: &[S],
    ) -> L {
        let node_f = &self.nodes[node_idx];

        let index_of_max = node_f
            .iter()
            .take(node_f.len() - 1)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Check if constant
        if index_of_max == node_f.len() - 2 {
            return L::from_op(&node_f[node_f.len() - 1].to_string(), vec![]).unwrap();
        }

        // check if variable
        if index_of_max >= L::operator_names().len() {
            let var_idx = index_of_max - L::operator_names().len();
            return L::from_op(variable_names[var_idx].as_ref(), children).unwrap();
        }

        L::from_op(L::operator_names()[index_of_max], children).unwrap()
    }
}

fn initialize_feature_vector<L: Language + MetaInfo, S>(variable_names: &[S]) -> Vec<f64> {
    vec![0f64; L::operator_names().len() + variable_names.len() + 2]
}

#[cfg(test)]
mod tests {
    use crate::trs::halide::HalideLang;

    use super::*;

    #[test]
    fn pytorch_inverse() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = GraphData::new(&expr, &variable_names, false).unwrap();
        let new_expr: RecExpr<HalideLang> = data.to_rec_expr(&variable_names);
        assert_eq!(expr, new_expr);
        assert_eq!(expr.to_string(), new_expr.to_string());
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = GraphData::new(&expr, &variable_names, false).unwrap();

        assert_eq!(
            data.nodes(),
            vec![
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ],
                vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 35.0
                ],
                vec![
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ],
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 5.0
                ],
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 17.0
                ]
            ]
        );
        assert_eq!(
            data.edges(),
            &[vec![0, 1, 1, 0, 4, 5, 5, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]]
        );
    }
}
