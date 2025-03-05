use egg::{FromOp, Id, RecExpr};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::prelude::*;

use serde::Serialize;
use thiserror::Error;

use crate::trs::{MetaInfo, SymbolInfo};

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
    name: String,
    #[pyo3(get)]
    arity: usize,
    #[pyo3(get)]
    nth_child: usize,
    #[pyo3(get)]
    dfs_order: usize,
    #[pyo3(get)]
    depth: usize,
    symbol: SymbolInfo,
}

impl Node {
    #[must_use]
    pub fn new(
        name: String,
        arity: usize,
        nth_child: usize,
        dfs_order: usize,
        depth: usize,
        symbol: SymbolInfo,
    ) -> Self {
        Self {
            name,
            arity,
            nth_child,
            dfs_order,
            depth,
            symbol,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Node {
    #[must_use]
    #[getter]
    pub fn id(&self) -> usize {
        self.symbol.id()
    }

    #[must_use]
    #[getter]
    pub fn value(&self) -> Option<String> {
        self.symbol.value()
    }
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct TreeData {
    #[pyo3(get)]
    nodes: Vec<Node>,
    #[pyo3(get)]
    adjacency: Vec<(usize, usize)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl TreeData {
    #[must_use]
    pub fn transposed_adjacency(&self) -> [Vec<usize>; 2] {
        let (a, b) = self.adjacency.iter().map(|x| x.to_owned()).unzip();
        [a, b]
    }

    #[must_use]
    pub fn count_symbols(&self, n_symbols: usize, n_vars: usize) -> Vec<usize> {
        let mut f = vec![0; n_symbols + n_vars];
        for n in &self.nodes {
            f[n.id()] += 1;
        }
        f
    }

    #[must_use]
    pub fn values(&self) -> Vec<String> {
        self.nodes.iter().filter_map(|n| n.symbol.value()).collect()
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.name.clone()).collect()
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

    fn feature_vec_to_node<L: MetaInfo + FromOp>(&self, node_idx: usize, children: Vec<Id>) -> L {
        // Ids go  | operators & metasymbols | consts | variables
        let node = &self.nodes[node_idx];

        if let Some(v) = node.symbol.value() {
            // If it is a constant we can safely unwrap
            L::from_op(&v, vec![]).unwrap()
        } else {
            L::from_op(L::operators()[node.id()], children).unwrap()
        }
    }
}

impl<L: MetaInfo + FromOp> TryFrom<&RecExpr<L>> for TreeData {
    type Error = TreeDataError;

    fn try_from(rec_expr: &RecExpr<L>) -> Result<TreeData, TreeDataError> {
        fn rec<L: MetaInfo + FromOp>(
            rec_expr: &RecExpr<L>,
            node: &L,
            graph_data: &mut TreeData,
            depth: usize,
            nth_node: usize,
        ) -> Result<usize, TreeDataError> {
            // All operators, variable_names, and last two are the constant symbol with its value
            let arity = node.children().len();
            let parent_idx = graph_data.add_node(Node::new(
                node.to_string(),
                arity,
                nth_node,
                graph_data.nodes.len(),
                depth,
                node.symbol_info(),
            ));
            for (nth_child, c_id) in node.children().iter().enumerate() {
                let child_idx = rec(rec_expr, &rec_expr[*c_id], graph_data, depth + 1, nth_child)?;
                graph_data.add_adjacency(parent_idx, child_idx);
            }
            Ok(parent_idx)
        }

        let root = rec_expr.root();
        // All operators, one for const, and variable_names
        let mut graph_data = TreeData {
            nodes: Vec::new(),
            adjacency: Vec::new(),
        };
        rec(rec_expr, &rec_expr[root], &mut graph_data, 0, 0)?;
        graph_data.adjacency.sort_unstable();
        Ok(graph_data)
    }
}

impl<L: MetaInfo + FromOp> TryFrom<&TreeData> for RecExpr<L> {
    type Error = TreeDataError;

    fn try_from(tree_data: &TreeData) -> Result<RecExpr<L>, TreeDataError> {
        fn rec<LL: MetaInfo + FromOp>(
            data: &TreeData,
            node_idx: usize,
            stack: &mut Vec<LL>,
        ) -> Result<Id, TreeDataError> {
            let children = data
                .adjacency
                .iter()
                .filter(|(p, _)| *p == node_idx)
                .map(|(_, c)| *c)
                .map(|child_idx| rec::<LL>(data, child_idx, stack))
                .collect::<Result<_, _>>()?;
            let node = data.feature_vec_to_node::<LL>(node_idx, children);
            stack.push(node);
            Ok(Id::from(stack.len() - 1))
        }

        // Only one node has nothing pointing to it: The root
        let (root, _) = tree_data
            .adjacency
            .iter()
            .find(|(p, _)| !tree_data.adjacency.iter().any(|(_, c)| p == c))
            .unwrap();

        let mut stack = vec![];
        rec(tree_data, *root, &mut stack)?;
        Ok(stack.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::trs::{SymbolType, halide::HalideLang};

    use egg::RecExpr;

    #[test]
    fn pytorch_inverse() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();

        let new_expr: RecExpr<HalideLang> = (&data).try_into().unwrap();
        assert_eq!(expr, new_expr);
        assert_eq!(expr.to_string(), new_expr.to_string());
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        assert_eq!(
            data.nodes,
            vec![
                Node {
                    name: "<".to_owned(),
                    arity: 2,
                    nth_child: 0,
                    dfs_order: 0,
                    depth: 0,
                    symbol: SymbolInfo::new(7, SymbolType::Operator)
                },
                Node {
                    name: "*".to_owned(),
                    arity: 2,
                    nth_child: 0,
                    dfs_order: 1,
                    depth: 1,
                    symbol: SymbolInfo::new(2, SymbolType::Operator)
                },
                Node {
                    name: "v0".to_owned(),
                    arity: 0,
                    nth_child: 0,
                    dfs_order: 2,
                    depth: 2,
                    symbol: SymbolInfo::new(18, SymbolType::Variable("v0".to_owned()))
                },
                Node {
                    name: "35".to_owned(),
                    arity: 0,
                    nth_child: 1,
                    dfs_order: 3,
                    depth: 2,
                    symbol: SymbolInfo::new(17, SymbolType::Constant("35".to_owned()))
                },
                Node {
                    name: "*".to_owned(),
                    arity: 2,
                    nth_child: 1,
                    dfs_order: 4,
                    depth: 1,
                    symbol: SymbolInfo::new(2, SymbolType::Operator)
                },
                Node {
                    name: "+".to_owned(),
                    arity: 2,
                    nth_child: 0,
                    dfs_order: 5,
                    depth: 2,
                    symbol: SymbolInfo::new(0, SymbolType::Operator)
                },
                Node {
                    name: "v0".to_owned(),
                    arity: 0,
                    nth_child: 0,
                    dfs_order: 6,
                    depth: 3,
                    symbol: SymbolInfo::new(18, SymbolType::Variable("v0".to_owned()))
                },
                Node {
                    name: "5".to_owned(),
                    arity: 0,
                    nth_child: 1,
                    dfs_order: 7,
                    depth: 3,
                    symbol: SymbolInfo::new(17, SymbolType::Constant("5".to_owned()))
                },
                Node {
                    name: "17".to_owned(),
                    arity: 0,
                    nth_child: 1,
                    dfs_order: 8,
                    depth: 2,
                    symbol: SymbolInfo::new(17, SymbolType::Constant("17".to_owned()))
                }
            ]
        );
        assert_eq!(
            data.transposed_adjacency(),
            [vec![0, 0, 1, 1, 4, 4, 5, 5], vec![1, 4, 2, 3, 5, 8, 6, 7]]
        );
    }

    // #[test]
    // fn unknown_symbol_false() {
    //     let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
    //     assert!(TreeData::new(&expr, false).is_err());
    // }

    // #[test]
    // fn unknown_symbol_true() {
    //     let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v1 5 ) 17 ) )".parse().unwrap();
    //     assert!(TreeData::new(&expr, true).is_ok());
    // }
}
