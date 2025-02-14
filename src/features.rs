use egg::{FromOp, Id, Language, RecExpr};
use serde::Serialize;
use thiserror::Error;

use crate::trs::MetaInfo;

pub trait AsFeatures<L: MetaInfo> {
    fn count_symbols<S: AsRef<str>>(
        &self,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<Vec<usize>, FeatureError>;
    fn arity(&self, position: usize) -> usize;

    fn size(&self) -> usize;

    fn depth(&self) -> usize;
}

impl<L: Language + MetaInfo> AsFeatures<L> for RecExpr<L> {
    fn count_symbols<S: AsRef<str>>(
        &self,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<Vec<usize>, FeatureError> {
        fn rec<L: Language + MetaInfo, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            ignore_unknown: bool,
            f: &mut Vec<usize>,
        ) -> Result<(), FeatureError> {
            let (idx, _) =
                node.symbol_type()
                    .to_idx(variable_names, L::n_non_vars(), ignore_unknown)?;
            f[idx] += 1;
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
        // Add space for ignored symbols to be counted
        let space_for_ignored = usize::from(ignore_unknown);
        let mut f = vec![0usize; L::n_non_vars() + variable_names.len() + space_for_ignored];
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
    nodes: Vec<usize>,
    edges: [Vec<usize>; 2],
    const_values: Vec<Option<f64>>,
    ignore_unknown: bool,
}

impl GraphData {
    pub fn nodes(&self) -> &[usize] {
        &self.nodes
    }

    pub fn edges(&self) -> &[Vec<usize>; 2] {
        &self.edges
    }

    pub fn const_values(&self) -> &[Option<f64>] {
        &self.const_values
    }

    pub fn ignore_unknown(&self) -> bool {
        self.ignore_unknown
    }

    pub fn new<L: MetaInfo, S: AsRef<str>>(
        rec_expr: &RecExpr<L>,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<GraphData, FeatureError> {
        fn rec<L: Language + MetaInfo, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            ignore_unknown: bool,
            nodes: &mut Vec<usize>,
            const_values: &mut Vec<Option<f64>>,
            edges: &mut [Vec<usize>; 2],
        ) -> Result<(), FeatureError> {
            // All operators, variable_names, and last two are the constant symbol with its value
            let (node_id, const_value) =
                node.symbol_type()
                    .to_idx(variable_names, L::n_non_vars(), ignore_unknown)?;
            nodes.push(node_id);
            const_values.push(const_value);
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
                    const_values,
                    edges,
                )?;
            }
            Ok(())
        }

        let root = rec_expr.root();
        // All operators, one for const, and variable_names
        let mut nodes = Vec::new();
        let mut edges = [Vec::new(), Vec::new()];
        let mut const_values = Vec::new();
        rec(
            rec_expr,
            &rec_expr[root],
            variable_names,
            ignore_unknown,
            &mut nodes,
            &mut const_values,
            &mut edges,
        )?;
        Ok(GraphData {
            nodes,
            edges,
            const_values,
            ignore_unknown,
        })
    }

    pub fn to_rec_expr<L: MetaInfo + FromOp, S: AsRef<str>>(
        &self,
        variable_names: &[S],
    ) -> Result<RecExpr<L>, FeatureError> {
        fn rec<LL: MetaInfo + FromOp, SS: AsRef<str>>(
            data: &GraphData,
            variable_names: &[SS],
            node_idx: usize,
            stack: &mut Vec<LL>,
        ) -> Result<Id, FeatureError> {
            let children = data.edges()[0]
                .iter()
                .zip(data.edges()[1].iter())
                .filter(|(p, _)| **p == node_idx)
                .map(|(_, c)| *c)
                .map(|child_idx| rec::<LL, SS>(data, variable_names, child_idx, stack))
                .collect::<Result<_, _>>()?;
            let node = data.feature_vec_to_node::<LL, SS>(node_idx, children, variable_names);
            stack.push(node);
            Ok(Id::from(stack.len() - 1))
        }

        if self.ignore_unknown {
            return Err(FeatureError::ImpossibleReconstruction);
        }

        // Only one node has nothing pointing to it: The root
        let root = self.edges[0]
            .iter()
            .find(|x| !self.edges[1].contains(x))
            .unwrap();

        let mut stack = vec![];
        rec(self, variable_names, *root, &mut stack)?;
        Ok(stack.into())
    }

    fn feature_vec_to_node<L: MetaInfo + FromOp, S: AsRef<str>>(
        &self,
        node_idx: usize,
        children: Vec<Id>,
        variable_names: &[S],
    ) -> L {
        // It goes consts | operators & metasymbols | variables
        let node_id = self.nodes[node_idx];

        if node_id < L::N_CONST_TYPES {
            // If it is a constant we can safely unwrap and stringify
            let const_value_string = self.const_values[node_idx].unwrap().to_string();
            L::from_op(&const_value_string, vec![]).unwrap()
        } else if node_id < L::n_non_vars() {
            L::from_op(L::named_symbols()[node_id - L::N_CONST_TYPES], children).unwrap()
        } else {
            // At the end are the variables so if it is lower we can just lookup
            let var_idx = node_id - L::n_non_vars();
            L::from_op(variable_names[var_idx].as_ref(), children).unwrap()
        }
    }

    pub fn num_node_types<L: MetaInfo, S: AsRef<str>>(
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> usize {
        // Plus 1 for constant!
        let base_len = L::n_non_vars() + variable_names.len();
        if ignore_unknown {
            // Additional unknown symbol
            base_len + 1
        } else {
            base_len
        }
    }
}

#[derive(Debug, Error)]
pub enum FeatureError {
    #[error("Symbol not in language: {0}")]
    UnknownSymbol(String),
    #[error("Cannot reconstruct ignored symbols")]
    ImpossibleReconstruction,
}

#[cfg(test)]
mod tests {
    use crate::trs::{halide::HalideLang, rise::RiseLang};

    use super::*;

    #[test]
    fn pytorch_inverse() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = GraphData::new(&expr, &variable_names, false).unwrap();

        let new_expr: RecExpr<HalideLang> = data.to_rec_expr(&variable_names).unwrap();
        assert_eq!(expr, new_expr);
        assert_eq!(expr.to_string(), new_expr.to_string());
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = GraphData::new(&expr, &variable_names, false).unwrap();

        assert_eq!(data.nodes(), vec![9, 4, 18, 1, 4, 2, 18, 1, 1]);
        assert_eq!(
            data.edges(),
            &[vec![0, 1, 1, 0, 4, 5, 5, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]]
        );
    }

    #[test]
    fn rise_num_node_types() {
        let num_native_symbols = RiseLang::named_symbols().len();
        let variable_names = vec!["v0", "v1"];
        let num_symbols = GraphData::num_node_types::<RiseLang, _>(&variable_names, true);
        // Plus 1 for ignored symbols
        assert_eq!(
            num_symbols,
            num_native_symbols + variable_names.len() + RiseLang::N_CONST_TYPES + 1
        );
    }

    #[test]
    fn unknown_symbol_false() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        assert!(GraphData::new(&expr, &variable_names, false).is_err());
    }

    #[test]
    fn unknown_symbol_true() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        assert!(GraphData::new(&expr, &variable_names, true).is_ok());
    }
}
