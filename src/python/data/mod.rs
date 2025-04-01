mod nodes;

use egg::{FromOp, Id, RecExpr};
use hashbrown::{HashMap, HashSet};
pub use nodes::Node;
pub use nodes::NodeOrPlaceHolder;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::prelude::*;

use serde::Serialize;
use thiserror::Error;

use crate::trs::MetaInfo;

#[derive(Debug, Error)]
pub enum TreeDataError {
    #[error("Symbol not in language: {0}")]
    UnknownSymbol(String),
    #[error("Cannot reconstruct ignored symbols")]
    ImpossibleReconstruction,
    #[error("Max arity reached while trying to parse partial term: {0}")]
    MaxArity(usize),
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct TreeData {
    nodes: Vec<NodeOrPlaceHolder>,
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

    /// Gives a matrix that describes the relationship of an ancestor to a child as a distance between them
    /// maximum distance (positive or negative) to be encoded mapped to the range 2 * max_rel_distance
    /// If the distance is too large or no relationship exists, 0 is returned
    #[must_use]
    #[pyo3(signature = (max_rel_distance, double_pad=true))]
    pub fn anc_matrix(&self, max_rel_distance: usize, double_pad: bool) -> Vec<Vec<usize>> {
        fn cmp_nodes(
            a: usize,
            b: usize,
            par_child: &HashMap<usize, HashSet<usize>>,
            curr_distance: usize,
        ) -> Option<usize> {
            par_child.get(&a).and_then(|cs| {
                cs.contains(&b).then_some(curr_distance).or_else(|| {
                    cs.iter()
                        .find_map(|c| cmp_nodes(*c, b, par_child, curr_distance + 1))
                })
            })
        }

        let par_child = self.adjacency.iter().fold(
            HashMap::new(),
            |mut par_child_table, (parent_idx, child_idx)| {
                par_child_table
                    .entry(*parent_idx)
                    .and_modify(|v: &mut HashSet<usize>| {
                        v.insert(*child_idx);
                    })
                    .or_insert(HashSet::from([*child_idx]));
                par_child_table
            },
        );

        let center = max_rel_distance + 1;

        let i = (0..self.adjacency.len()).map(|a_idx| {
            let inner_i = (0..self.adjacency.len()).map(|b_idx| {
                if a_idx == b_idx {
                    return center; // Distance to self is always 0
                }
                if let Some(d) = cmp_nodes(a_idx, b_idx, &par_child, 1) {
                    (d < max_rel_distance).then_some(center + d)
                // Positive since parent to child
                } else if let Some(d) = cmp_nodes(b_idx, a_idx, &par_child, 1) {
                    (d < max_rel_distance).then_some(center - d)
                // Negative since child to parent
                } else {
                    None
                }
                .unwrap_or(0)
                // If no connection => inf
            });
            if !double_pad {
                return inner_i.collect();
            }
            let mut r_inner = vec![0];
            r_inner.extend(inner_i);
            r_inner.push(0);
            r_inner
        });
        if !double_pad {
            return i.collect();
        }
        let mut r = vec![vec![0; self.adjacency.len() + 2]];
        r.extend(i);
        r.push(vec![0; self.adjacency.len() + 2]);
        r
    }

    /// Gives a matrix that describes the sibling relationship in nodes
    /// max_relative_distance describes the maximum distance (positive or negative) to be encoded,
    /// mapped to the range 2 * max_relative_distance
    /// If the distance is too large or no relationship exists, 0 is returned
    #[must_use]
    #[pyo3(signature = (max_rel_distance, double_pad=true))]
    pub fn sib_matrix(&self, max_rel_distance: usize, double_pad: bool) -> Vec<Vec<usize>> {
        fn cmp_nodes(
            a: usize,
            b: usize,
            par_child: &HashMap<usize, Vec<usize>>,
            child_par: &HashMap<usize, usize>,
            max_relative_distance: usize,
            center: usize,
        ) -> Option<usize> {
            // Distance to self is always 0 aka center
            // This catches the special case where root is compared to root
            // which would be problematic in the if let since root has no parents
            if a == b {
                return Some(center);
            }

            // Root case where a and b are both root and have no parents is caught by a==b
            if let (Some(par_idx_a), Some(par_idx_b)) = (child_par.get(&a), child_par.get(&b)) {
                // Sibling distance only makes sense if both have the same direct parent, otherwise infinite distance
                if par_idx_a != par_idx_b {
                    return None;
                }
                // If in child_par_map it must be in par_child_map
                let sibilings = par_child.get(par_idx_a).unwrap();
                let pos_a = sibilings.iter().position(|x| x == &a).unwrap();
                let pos_b = sibilings.iter().position(|x| x == &b).unwrap();
                let d = usize::abs_diff(pos_a, pos_b);

                if d >= max_relative_distance {
                    return None;
                }
                // == case caught earlier

                if pos_a < pos_b {
                    Some(center + d)
                } else {
                    Some(center - d)
                }
            } else {
                None // Either not related or bigger distance than max so we return max
            }
        }

        let center = max_rel_distance + 1;

        let (par_child, child_par) = self.adjacency.iter().fold(
            (HashMap::new(), HashMap::new()),
            |(mut par_child, mut child_par), (parent, child)| {
                child_par.insert(*child, *parent);
                par_child
                    .entry(*parent)
                    .and_modify(|v: &mut Vec<usize>| {
                        v.push(*child);
                    })
                    .or_insert(vec![*child]);
                (par_child, child_par)
            },
        );

        let i = (0..self.adjacency.len()).map(|a_idx| {
            let inner_i = (0..self.adjacency.len()).map(|b_idx| {
                cmp_nodes(
                    a_idx,
                    b_idx,
                    &par_child,
                    &child_par,
                    max_rel_distance,
                    center,
                )
                .unwrap_or(0)
            });
            if !double_pad {
                return inner_i.collect();
            }
            let mut r_inner = vec![0];
            r_inner.extend(inner_i);
            r_inner.push(0);
            r_inner
        });
        if double_pad {
            let mut r = vec![vec![0; self.adjacency.len() + 2]];
            r.extend(i);
            r.push(vec![0; self.adjacency.len() + 2]);
            r
        } else {
            i.collect()
        }
    }

    #[must_use]
    pub fn count_symbols(&self, n_symbols: usize, n_vars: usize) -> Vec<usize> {
        let mut f = vec![0; n_symbols + n_vars];
        for n in &self.nodes {
            if let NodeOrPlaceHolder::Node(node) = n {
                f[node.id()] += 1;
            }
        }
        f
    }

    #[must_use]
    pub fn values(&self) -> Vec<String> {
        self.nodes.iter().filter_map(|n| n.value()).collect()
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.name().clone()).collect()
    }

    fn arity(&self, position: usize) -> Option<usize> {
        self.nodes[position].arity()
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|x| x.depth()).max().unwrap()
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

    fn add_node(&mut self, node: NodeOrPlaceHolder) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn feature_vec_to_node<L: MetaInfo + FromOp>(
        &self,
        node_idx: usize,
        children: Vec<Id>,
    ) -> Result<L, TreeDataError> {
        // Ids go  | operators & metasymbols | consts | variables
        let NodeOrPlaceHolder::Node(node) = &self.nodes[node_idx] else {
            return Err(TreeDataError::ImpossibleReconstruction);
        };

        if let Some(v) = node.symbol_info().value() {
            // If it is a constant we can safely unwrap
            Ok(L::from_op(&v, vec![]).unwrap())
        } else {
            Ok(L::from_op(L::operators()[node.id()], children).unwrap())
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
            let parent_idx = graph_data.add_node(NodeOrPlaceHolder::Node(Node::new(
                node.to_string(),
                arity,
                nth_node,
                graph_data.nodes.len(),
                depth,
                node.symbol_info(),
            )));
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

/// Tries to parse a list of tokens into a list of nodes in the language
///
/// # Errors
///
/// This function will return an error if it cannot be parsed as a partial term
pub fn partial_parse<L: FromOp + MetaInfo>(
    mut token_list: Vec<String>,
) -> Result<Vec<Option<L>>, TreeDataError> {
    token_list.reverse();
    let max_arity = 1233;

    let mut children_ids = Vec::new();
    let mut nodes = Vec::new();
    for token in &token_list {
        // Either we are parsing a parent, then we take all the children currently on the stack
        if let Ok(node) = L::from_op(token, children_ids.clone()) {
            nodes.push(Some(node));
            children_ids.clear();
            children_ids.push(Id::from(nodes.len() - 1));
        // Or we are parsing a sibling child with no children, so we put it on the stack
        } else if let Ok(node) = L::from_op(token, Vec::new()) {
            nodes.push(Some(node));
            children_ids.push(Id::from(nodes.len() - 1));
        // Or we are parsing and incomplete parent that only has some children already generated
        } else {
            for n in 0..max_arity {
                if let Ok(node) = L::from_op(token, children_ids.clone()) {
                    nodes.push(Some(node));
                    children_ids.clear();
                    children_ids.push(Id::from(nodes.len() - 1));
                    break;
                }
                nodes.push(None);
                children_ids.push(Id::from(nodes.len() - 1));
                if n > max_arity {
                    return Err(TreeDataError::MaxArity(n));
                }
            }
        }
    }
    Ok(nodes)
}

impl<L: MetaInfo + FromOp> TryFrom<Vec<Option<L>>> for TreeData {
    type Error = TreeDataError;

    fn try_from(node_list: Vec<Option<L>>) -> Result<TreeData, TreeDataError> {
        fn rec<L: MetaInfo + FromOp>(
            node_list: &[Option<L>],
            node: Option<&L>,
            graph_data: &mut TreeData,
            depth: usize,
            position: usize,
        ) -> Result<usize, TreeDataError> {
            let Some(actual_l) = node else {
                return Ok(graph_data.add_node(NodeOrPlaceHolder::Placeholder {
                    depth,
                    dfs_order: graph_data.nodes.len(),
                    nth_child: position,
                }));
            };

            // All operators, variable_names, and last two are the constant symbol with its value
            let arity = actual_l.children().len();
            let parent_idx = graph_data.add_node(NodeOrPlaceHolder::Node(Node::new(
                actual_l.to_string(),
                arity,
                position,
                graph_data.nodes.len(),
                depth,
                actual_l.symbol_info(),
            )));
            for (i, c_id) in actual_l.children().iter().enumerate() {
                let child_idx = rec(
                    node_list,
                    node_list[usize::from(*c_id)].as_ref(),
                    graph_data,
                    depth + 1,
                    i,
                )?;
                graph_data.add_adjacency(parent_idx, child_idx);
            }
            Ok(parent_idx)
        }

        let root = node_list.len();
        // All operators, one for const, and variable_names
        let mut graph_data = TreeData {
            nodes: Vec::new(),
            adjacency: Vec::new(),
        };
        rec(&node_list, node_list[root].as_ref(), &mut graph_data, 0, 0)?;
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
            let node = data.feature_vec_to_node::<LL>(node_idx, children)?;
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

    use crate::trs::{SymbolInfo, SymbolType, halide::HalideLang};

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
    fn sib_matrix() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        let par_sib = data.sib_matrix(15, false);

        assert_eq!(
            par_sib,
            vec![
                vec![16, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 16, 0, 0, 17, 0, 0, 0],
                vec![0, 0, 16, 17, 0, 0, 0, 0],
                vec![0, 0, 15, 16, 0, 0, 0, 0],
                vec![0, 15, 0, 0, 16, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 16, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 16, 17],
                vec![0, 0, 0, 0, 0, 0, 15, 16]
            ]
        );
    }

    #[test]
    fn anc_matrix() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        let par_sib = data.anc_matrix(15, false);

        assert_eq!(
            par_sib,
            vec![
                vec![16, 17, 18, 18, 17, 18, 19, 19],
                vec![15, 16, 17, 17, 0, 0, 0, 0],
                vec![14, 15, 16, 0, 0, 0, 0, 0],
                vec![14, 15, 0, 16, 0, 0, 0, 0],
                vec![15, 0, 0, 0, 16, 17, 18, 18],
                vec![14, 0, 0, 0, 15, 16, 17, 17],
                vec![13, 0, 0, 0, 14, 15, 16, 0],
                vec![13, 0, 0, 0, 14, 15, 0, 16]
            ]
        );
    }

    #[test]
    fn anc_matrix_padded() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        let par_sib = data.anc_matrix(15, true);

        assert_eq!(
            par_sib,
            vec![
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 16, 17, 18, 18, 17, 18, 19, 19, 0],
                vec![0, 15, 16, 17, 17, 0, 0, 0, 0, 0],
                vec![0, 14, 15, 16, 0, 0, 0, 0, 0, 0],
                vec![0, 14, 15, 0, 16, 0, 0, 0, 0, 0],
                vec![0, 15, 0, 0, 0, 16, 17, 18, 18, 0],
                vec![0, 14, 0, 0, 0, 15, 16, 17, 17, 0],
                vec![0, 13, 0, 0, 0, 14, 15, 16, 0, 0],
                vec![0, 13, 0, 0, 0, 14, 15, 0, 16, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn sib_matrix_padded() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        let par_sib = data.sib_matrix(15, true);

        assert_eq!(
            par_sib,
            vec![
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 16, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 16, 0, 0, 17, 0, 0, 0, 0],
                vec![0, 0, 0, 16, 17, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 15, 16, 0, 0, 0, 0, 0],
                vec![0, 0, 15, 0, 0, 16, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 16, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 16, 17, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 15, 16, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).try_into().unwrap();
        assert_eq!(
            data.nodes,
            vec![
                NodeOrPlaceHolder::Node(Node::new(
                    "<".to_owned(),
                    2,
                    0,
                    0,
                    0,
                    SymbolInfo::new(7, SymbolType::Operator)
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "*".to_owned(),
                    2,
                    0,
                    1,
                    1,
                    SymbolInfo::new(2, SymbolType::Operator)
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "v0".to_owned(),
                    0,
                    0,
                    2,
                    2,
                    SymbolInfo::new(18, SymbolType::Variable("v0".to_owned()))
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "35".to_owned(),
                    0,
                    1,
                    3,
                    2,
                    SymbolInfo::new(17, SymbolType::Constant("35".to_owned()))
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "*".to_owned(),
                    2,
                    1,
                    4,
                    1,
                    SymbolInfo::new(2, SymbolType::Operator)
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "+".to_owned(),
                    2,
                    0,
                    5,
                    2,
                    SymbolInfo::new(0, SymbolType::Operator)
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "v0".to_owned(),
                    0,
                    0,
                    6,
                    3,
                    SymbolInfo::new(18, SymbolType::Variable("v0".to_owned()))
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "5".to_owned(),
                    0,
                    1,
                    7,
                    3,
                    SymbolInfo::new(17, SymbolType::Constant("5".to_owned()))
                )),
                NodeOrPlaceHolder::Node(Node::new(
                    "17".to_owned(),
                    0,
                    1,
                    8,
                    2,
                    SymbolInfo::new(17, SymbolType::Constant("17".to_owned()))
                ))
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
