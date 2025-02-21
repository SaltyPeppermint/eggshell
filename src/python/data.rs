use egg::{FromOp, Id, RecExpr};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use rayon::prelude::*;

use serde::Serialize;
use thiserror::Error;

use crate::trs::{MetaInfo, SymbolType};

#[derive(Debug, Error)]
pub enum TreeDataError {
    #[error("Symbol not in language: {0}")]
    UnknownSymbol(String),
    #[error("Cannot reconstruct ignored symbols")]
    ImpossibleReconstruction,
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct Node {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    arity: usize,
    #[pyo3(get)]
    nth_child: usize,
    #[pyo3(get)]
    dfs_order: usize,
    #[pyo3(get)]
    depth: usize,
    #[pyo3(get)]
    const_value: Option<f64>,
}

impl Node {
    #[must_use]
    pub fn new(
        id: usize,
        name: String,
        arity: usize,
        nth_child: usize,
        dfs_order: usize,
        depth: usize,
        const_value: Option<f64>,
    ) -> Self {
        Self {
            id,
            name,
            arity,
            nth_child,
            dfs_order,
            depth,
            const_value,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct TreeData {
    nodes: Vec<Node>,
    adjacency: Vec<(usize, usize)>,
    ignore_unknown: bool,
}

#[pymethods]
impl TreeData {
    #[must_use]
    pub fn nodes(&self) -> Vec<Node> {
        self.nodes.clone()
    }

    #[must_use]
    pub fn adjacency(&self) -> &[(usize, usize)] {
        &self.adjacency
    }

    #[must_use]
    pub fn transposed_adjacency(&self) -> [Vec<usize>; 2] {
        let (a, b) = self.adjacency.iter().map(|x| x.to_owned()).unzip();
        [a, b]
    }

    #[must_use]
    pub fn ignore_unknown(&self) -> bool {
        self.ignore_unknown
    }

    #[must_use]
    pub fn count_symbols(&self, n_symbols: usize, n_vars: usize) -> Vec<usize> {
        let mut f = vec![0; n_symbols + n_vars];
        for n in &self.nodes {
            f[n.id] += 1;
        }
        f
    }

    fn arity(&self, position: usize) -> usize {
        self.nodes[position].arity
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|x| x.depth).max().unwrap()
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    #[expect(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn simple_feature_names(
        &self,
        symbol_names: Vec<String>,
        var_names: Vec<String>,
    ) -> Vec<String> {
        let mut s = symbol_names.clone();
        if self.ignore_unknown {
            s.push("IGNORED".to_owned());
        }
        s.push("CONSTANT".to_owned());
        s.extend(var_names);
        s.push("SIZE".to_owned());
        s.push("DEPTH".to_owned());
        s
    }

    #[expect(clippy::cast_precision_loss)]
    #[must_use]
    pub fn simple_features(&self, n_symbols: usize, n_vars: usize) -> Vec<f64> {
        let mut features = self.count_symbols(n_symbols, n_vars);
        features.push(self.size());
        features.push(self.depth());
        features.into_iter().map(|v| v as f64).collect()
    }

    #[expect(clippy::needless_pass_by_value)]
    #[must_use]
    #[staticmethod]
    pub fn batch_simple_features(
        tree_datas: Vec<TreeData>,
        n_symbols: usize,
        n_vars: usize,
    ) -> Vec<Vec<f64>> {
        tree_datas
            .par_iter()
            .map(|d| d.simple_features(n_symbols, n_vars))
            .collect::<Vec<_>>()
    }
}

impl TreeData {
    fn add_adjacency(&mut self, parent: usize, child: usize) {
        self.adjacency.push((parent, child));
    }

    fn add_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    #[expect(clippy::missing_errors_doc)]
    pub fn new<L: MetaInfo + FromOp, S: AsRef<str>>(
        rec_expr: &RecExpr<L>,
        variable_names: &[S],
        ignore_unknown: bool,
    ) -> Result<TreeData, TreeDataError> {
        fn rec<L: MetaInfo + FromOp, S: AsRef<str>>(
            rec_expr: &RecExpr<L>,
            node: &L,
            variable_names: &[S],
            graph_data: &mut TreeData,
            depth: usize,
            nth_child: usize,
        ) -> Result<usize, TreeDataError> {
            // All operators, variable_names, and last two are the constant symbol with its value
            let (node_id, const_value) = match node.symbol_type() {
                SymbolType::Operator(idx) | SymbolType::MetaSymbol(idx) => (idx, None),
                // right behind the operators len for the constant type
                SymbolType::Constant(idx, v) => (idx, Some(v)),
                SymbolType::Variable(name) => {
                    if let Some(var_idx) = variable_names.iter().position(|x| x.as_ref() == name) {
                        (L::n_non_vars() + var_idx, None)
                    } else if graph_data.ignore_unknown {
                        // Ignored symbol
                        (L::n_non_vars() + variable_names.len(), None)
                    } else {
                        return Err(TreeDataError::UnknownSymbol(name.to_owned()));
                    }
                }
            };

            let arity = node.children().len();
            let parent_idx = graph_data.add_node(Node {
                id: node_id,
                name: node.to_string(),
                arity,
                dfs_order: graph_data.nodes.len(),
                depth,
                nth_child,
                const_value,
            });
            for (i, c_id) in node.children().iter().enumerate() {
                let child_idx = rec(
                    rec_expr,
                    &rec_expr[*c_id],
                    variable_names,
                    graph_data,
                    depth + 1,
                    i,
                )?;
                graph_data.add_adjacency(parent_idx, child_idx);
            }
            Ok(parent_idx)
        }

        let root = rec_expr.root();
        // All operators, one for const, and variable_names
        let mut graph_data = TreeData {
            nodes: Vec::new(),
            adjacency: Vec::new(),
            ignore_unknown,
        };
        rec(
            rec_expr,
            &rec_expr[root],
            variable_names,
            &mut graph_data,
            0,
            0,
        )?;
        graph_data.adjacency.sort_unstable();
        Ok(graph_data)
    }

    #[expect(clippy::missing_panics_doc, clippy::missing_errors_doc)]
    pub fn to_rec_expr<L: MetaInfo + FromOp, S: AsRef<str>>(
        &self,
        variable_names: &[S],
    ) -> Result<RecExpr<L>, TreeDataError> {
        fn rec<LL: MetaInfo + FromOp, SS: AsRef<str>>(
            data: &TreeData,
            variable_names: &[SS],
            node_idx: usize,
            stack: &mut Vec<LL>,
        ) -> Result<Id, TreeDataError> {
            let children = data
                .adjacency()
                .iter()
                .filter(|(p, _)| *p == node_idx)
                .map(|(_, c)| *c)
                .map(|child_idx| rec::<LL, SS>(data, variable_names, child_idx, stack))
                .collect::<Result<_, _>>()?;
            let node = data.feature_vec_to_node::<LL, SS>(node_idx, children, variable_names);
            stack.push(node);
            Ok(Id::from(stack.len() - 1))
        }

        if self.ignore_unknown {
            return Err(TreeDataError::ImpossibleReconstruction);
        }

        // Only one node has nothing pointing to it: The root
        let (root, _) = self
            .adjacency
            .iter()
            .find(|(p, _)| !self.adjacency.iter().any(|(_, c)| p == c))
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
        // Ids go consts | operators & metasymbols | variables
        let node = &self.nodes[node_idx];

        if node.id < L::N_CONST_TYPES {
            // If it is a constant we can safely unwrap and stringify
            let const_value_string = node.const_value.unwrap().to_string();
            L::from_op(&const_value_string, vec![]).unwrap()
        } else if node.id < L::n_non_vars() {
            L::from_op(L::named_symbols()[node.id - L::N_CONST_TYPES], children).unwrap()
        } else {
            // At the end are the variables so if it is lower we can just lookup
            let var_idx = node.id - L::n_non_vars();
            L::from_op(variable_names[var_idx].as_ref(), children).unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::trs::halide::HalideLang;

    use egg::RecExpr;

    #[test]
    fn pytorch_inverse() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = TreeData::new(&expr, &variable_names, false).unwrap();

        let new_expr: RecExpr<HalideLang> = data.to_rec_expr(&variable_names).unwrap();
        assert_eq!(expr, new_expr);
        assert_eq!(expr.to_string(), new_expr.to_string());
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        let data = TreeData::new(&expr, &variable_names, false).unwrap();

        assert_eq!(
            data.nodes(),
            vec![
                Node::new(9, "<".to_owned(), 2, 0, 0, 0, None),
                Node::new(4, "*".to_owned(), 2, 0, 1, 1, None),
                Node::new(18, "v0".to_owned(), 0, 0, 2, 2, None),
                Node::new(1, "35".to_owned(), 0, 1, 3, 2, Some(35.0)),
                Node::new(4, "*".to_owned(), 2, 1, 4, 1, None),
                Node::new(2, "+".to_owned(), 2, 0, 5, 2, None),
                Node::new(18, "v0".to_owned(), 0, 0, 6, 3, None),
                Node::new(1, "5".to_owned(), 0, 1, 7, 3, Some(5.0)),
                Node::new(1, "17".to_owned(), 0, 1, 8, 2, Some(17.0))
            ]
        );
        assert_eq!(
            data.transposed_adjacency(),
            [vec![0, 0, 1, 1, 4, 4, 5, 5], vec![1, 4, 2, 3, 5, 8, 6, 7]]
        );
    }

    #[test]
    fn unknown_symbol_false() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        assert!(TreeData::new(&expr, &variable_names, false).is_err());
    }

    #[test]
    fn unknown_symbol_true() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
        let variable_names = vec!["v0"];
        assert!(TreeData::new(&expr, &variable_names, true).is_ok());
    }
}
