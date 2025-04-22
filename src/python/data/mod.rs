mod nodes;

use std::collections::VecDeque;

use egg::{FromOp, Id, RecExpr};
use hashbrown::{HashMap, HashSet};
pub use nodes::Node;
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
}

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct TreeData {
    // Remember, this is BFS traversal order
    nodes: Vec<Node>,
    adjacency_pairs: Vec<(usize, usize)>,
}

impl TreeData {
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl TreeData {
    #[must_use]
    pub fn transposed_adjacency(&self) -> [Vec<usize>; 2] {
        let (a, b) = self.adjacency_pairs.iter().map(|x| x.to_owned()).unzip();
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

        let par_child = self.adjacency_pairs.iter().fold(
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

        let i = (0..self.len()).map(|a_idx| {
            let inner_i = (0..self.len()).map(|b_idx| {
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
        let mut r = vec![vec![0; self.len() + 2]];
        r.extend(i);
        r.push(vec![0; self.len() + 2]);
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

        let (par_child, child_par) = self.adjacency_pairs.iter().fold(
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

        let i = (0..self.len()).map(|a_idx| {
            let inner_i = (0..self.len()).map(|b_idx| {
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
            let mut r = vec![vec![0; self.len() + 2]];
            r.extend(i);
            r.push(vec![0; self.len() + 2]);
            r
        } else {
            i.collect()
        }
    }

    #[must_use]
    pub fn count_symbols(&self, n_symbols: usize, n_vars: usize) -> Vec<usize> {
        let mut f = vec![0; n_symbols + n_vars];
        for node in &self.nodes {
            f[node.id()] += 1;
        }
        f
    }

    #[must_use]
    pub fn nodes(&self) -> Vec<Node> {
        self.nodes.iter().map(|n| n.to_owned()).collect()
    }

    #[must_use]
    pub fn values(&self) -> Vec<String> {
        self.nodes.iter().filter_map(|n| n.value()).collect()
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.name().clone()).collect()
    }

    fn arity(&self, position: usize) -> usize {
        self.nodes[position].arity()
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|x| x.depth()).max().unwrap()
    }

    #[must_use]
    pub fn __len__(&self) -> usize {
        self.len()
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
        features.push(self.len());
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
    fn feature_vec_to_node<L: MetaInfo + FromOp>(
        &self,
        node_idx: usize,
        children: Vec<Id>,
    ) -> Result<L, L::Error> {
        // Ids go  | operators & metasymbols | consts | variables
        let node = &self.nodes[node_idx];

        if let Some(v) = node.symbol_info().value() {
            // If it is a constant we can safely unwrap
            Ok(L::from_op(&v, vec![]).unwrap())
        } else {
            L::from_op(L::operators()[node.id()], children)
        }
    }
}

impl<L: MetaInfo + FromOp> From<&RecExpr<L>> for TreeData {
    fn from(rec_expr: &RecExpr<L>) -> TreeData {
        let mut nodes = Vec::new(); // Stores original indices in BFS order
        let mut adjacency_pairs = Vec::new();
        let mut queue = VecDeque::new();

        // Start with root node
        queue.push_back((rec_expr.root(), 0, 0, None)); // (original id, nth_node, depth, parent's BFS position)

        while let Some((original_idx, nth_node, depth, parent_bfs_pos)) = queue.pop_front() {
            // Record this node's position in BFS order
            let current_bfs_pos = nodes.len();
            let l = &rec_expr[original_idx];
            let new_node = Node::new(
                l.to_string(),
                l.children().len(),
                nth_node,
                nodes.len(),
                depth,
                l.symbol_info(),
            );
            nodes.push(new_node);

            // If this isn't the root node, record the adjacency pair
            if let Some(parent_pos) = parent_bfs_pos {
                adjacency_pairs.push((parent_pos, current_bfs_pos));
            }

            // Enqueue children with their parent's BFS position
            for (nth, children_id) in rec_expr[original_idx].children().iter().enumerate() {
                queue.push_back((*children_id, nth, depth + 1, Some(current_bfs_pos)));
            }
        }
        adjacency_pairs.sort_unstable();
        TreeData {
            nodes,
            adjacency_pairs,
        }
    }
}

impl<L: MetaInfo + FromOp> TryFrom<&TreeData> for RecExpr<L> {
    type Error = L::Error;

    fn try_from(tree_data: &TreeData) -> Result<RecExpr<L>, L::Error> {
        fn rec<LL: MetaInfo + FromOp>(
            data: &TreeData,
            node_idx: usize,
            stack: &mut Vec<LL>,
        ) -> Result<Id, LL::Error> {
            let children = data
                .adjacency_pairs
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
            .adjacency_pairs
            .iter()
            .find(|(p, _)| !tree_data.adjacency_pairs.iter().any(|(_, c)| p == c))
            .unwrap();

        let mut stack = vec![];
        rec(tree_data, *root, &mut stack)?;
        Ok(stack.into())
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use super::*;
    use crate::trs::{SymbolInfo, SymbolType, halide::HalideLang};

    #[test]
    fn dfs_order() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();

        assert_eq!(data.nodes[0].name(), "<");
        assert_eq!(data.nodes[1].name(), "*");
        assert_eq!(data.nodes[2].name(), "*");
        assert_eq!(data.nodes[3].name(), "[variable]");
        assert_eq!(data.nodes[4].name(), "[constant]");
        assert_eq!(data.nodes[5].name(), "+");
    }
    #[test]

    fn pytorch_inverse() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();

        let new_expr: RecExpr<HalideLang> = (&data).try_into().unwrap();
        assert_eq!(expr, new_expr);
        assert_eq!(expr.to_string(), new_expr.to_string());
    }

    #[test]
    fn sib_matrix() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();
        let par_sib = data.sib_matrix(15, false);

        assert_eq!(
            par_sib,
            vec![
                [16, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 16, 17, 0, 0, 0, 0, 0, 0],
                [0, 15, 16, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 16, 17, 0, 0, 0, 0],
                [0, 0, 0, 15, 16, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 16, 17, 0, 0],
                [0, 0, 0, 0, 0, 15, 16, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 16, 17],
                [0, 0, 0, 0, 0, 0, 0, 15, 16]
            ]
        );
    }

    #[test]
    fn anc_matrix() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();
        let par_anc = data.anc_matrix(15, false);

        assert_eq!(
            par_anc,
            vec![
                [16, 17, 17, 18, 18, 18, 18, 19, 19],
                [15, 16, 0, 17, 17, 0, 0, 0, 0],
                [15, 0, 16, 0, 0, 17, 17, 18, 18],
                [14, 15, 0, 16, 0, 0, 0, 0, 0],
                [14, 15, 0, 0, 16, 0, 0, 0, 0],
                [14, 0, 15, 0, 0, 16, 0, 17, 17],
                [14, 0, 15, 0, 0, 0, 16, 0, 0],
                [13, 0, 14, 0, 0, 15, 0, 16, 0],
                [13, 0, 14, 0, 0, 15, 0, 0, 16]
            ]
        );
    }

    #[test]
    fn anc_matrix_padded() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();
        let par_anc = data.anc_matrix(15, true);

        assert_eq!(
            par_anc,
            vec![
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 16, 17, 17, 18, 18, 18, 18, 19, 19, 0],
                [0, 15, 16, 0, 17, 17, 0, 0, 0, 0, 0],
                [0, 15, 0, 16, 0, 0, 17, 17, 18, 18, 0],
                [0, 14, 15, 0, 16, 0, 0, 0, 0, 0, 0],
                [0, 14, 15, 0, 0, 16, 0, 0, 0, 0, 0],
                [0, 14, 0, 15, 0, 0, 16, 0, 17, 17, 0],
                [0, 14, 0, 15, 0, 0, 0, 16, 0, 0, 0],
                [0, 13, 0, 14, 0, 0, 15, 0, 16, 0, 0],
                [0, 13, 0, 14, 0, 0, 15, 0, 0, 16, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn sib_matrix_padded() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();
        let par_sib = data.sib_matrix(15, true);

        assert_eq!(
            par_sib,
            vec![
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 16, 17, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 16, 17, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 16, 17, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 15, 16, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 15, 16, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn pytorch_format() {
        let expr: RecExpr<HalideLang> = "( < ( * v0 35 ) ( * ( + v0 5 ) 17 ) )".parse().unwrap();
        let data: TreeData = (&expr).into();
        assert_eq!(
            data.nodes(),
            vec![
                Node::new(
                    "<".into(),
                    2,
                    0,
                    0,
                    0,
                    SymbolInfo::new(7, SymbolType::Operator)
                ),
                Node::new(
                    "*".into(),
                    2,
                    0,
                    1,
                    1,
                    SymbolInfo::new(2, SymbolType::Operator)
                ),
                Node::new(
                    "*".into(),
                    2,
                    1,
                    2,
                    1,
                    SymbolInfo::new(2, SymbolType::Operator)
                ),
                Node::new(
                    "v0".into(),
                    0,
                    0,
                    3,
                    2,
                    SymbolInfo::new(18, SymbolType::Variable("v0".into()))
                ),
                Node::new(
                    "35".into(),
                    0,
                    1,
                    4,
                    2,
                    SymbolInfo::new(17, SymbolType::Constant("35".into()))
                ),
                Node::new(
                    "+".into(),
                    2,
                    0,
                    5,
                    2,
                    SymbolInfo::new(0, SymbolType::Operator)
                ),
                Node::new(
                    "17".into(),
                    0,
                    1,
                    6,
                    2,
                    SymbolInfo::new(17, SymbolType::Constant("17".into()))
                ),
                Node::new(
                    "v0".into(),
                    0,
                    0,
                    7,
                    3,
                    SymbolInfo::new(18, SymbolType::Variable("v0".into()))
                ),
                Node::new(
                    "5".into(),
                    0,
                    1,
                    8,
                    3,
                    SymbolInfo::new(17, SymbolType::Constant("5".into()))
                )
            ]
        );
        assert_eq!(
            data.transposed_adjacency(),
            [vec![0, 0, 1, 1, 2, 2, 5, 5], vec![1, 2, 3, 4, 5, 6, 7, 8]]
        );
    }
}
