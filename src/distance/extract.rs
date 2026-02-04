//! Tree extraction and minimum distance search for E-Graphs.
//!
//! This module provides iterators for enumerating trees from an e-graph
//! and functions for finding the tree with minimum edit distance to a reference.

use std::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use indicatif::ParallelProgressIterator;
use rayon::iter::{ParallelBridge, ParallelIterator};

use super::graph::EGraph;
use super::ids::{EClassId, ExprChildId};
use super::nodes::Label;
use super::str::EulerString;
use super::structural::structural_diff;
use super::tree::TreeNode;
use super::zs::{EditCosts, PreprocessedTree, tree_distance_with_ref};

/// Iterator that yields choice vectors without materializing trees.
/// Each choice vector can later be used with `tree_from_choices` to get the actual tree.
#[derive(Debug)]
pub struct ChoiceIter<'a, L: Label> {
    choices: Vec<usize>,
    path: PathTracker,
    egraph: &'a EGraph<L>,
}

impl<'a, L: Label> ChoiceIter<'a, L> {
    #[must_use]
    pub fn new(egraph: &'a EGraph<L>, max_revisits: usize) -> Self {
        Self {
            choices: Vec::new(),
            path: PathTracker::new(max_revisits),
            egraph,
        }
    }

    /// Find the next valid choice vector, modifying `choices` in place.
    ///
    /// If `choices` is empty or shorter than needed, finds the first valid tree.
    /// If `choices` already represents a tree and `advance` is true, finds the
    /// lexicographically next one.
    ///
    /// Returns `Some(last_idx)` on success, `None` if no more trees exist.
    fn next_choices(&mut self, id: EClassId, choice_idx: usize, advance: bool) -> Option<usize> {
        if !self.path.can_visit(id) {
            return None;
        }

        let class = self.egraph.class(id);

        // Determine starting node and whether to advance children
        let (start_node, advance_children) = if let Some(&c) = self.choices.get(choice_idx) {
            (c, advance)
        } else {
            self.choices.push(0);
            (0, false)
        };

        self.path.enter(id);

        let result =
            class
                .nodes()
                .iter()
                .enumerate()
                .skip(start_node)
                .find_map(|(node_idx, node)| {
                    self.choices[choice_idx] = node_idx;
                    let should_advance = advance_children && node_idx == start_node;

                    self.next_choices_children(node.children(), choice_idx, should_advance)
                        .or_else(|| {
                            self.choices.truncate(choice_idx + 1);
                            None
                        })
                });

        self.path.leave(id);
        result
    }

    /// Process children, optionally advancing to find the next combination.
    fn next_choices_children(
        &mut self,
        children: &[ExprChildId],
        parent_idx: usize,
        advance: bool,
    ) -> Option<usize> {
        let eclass_children: Vec<_> = children
            .iter()
            .filter_map(|c| match c {
                ExprChildId::EClass(id) => Some(*id),
                _ => None,
            })
            .collect();

        match (eclass_children.is_empty(), advance) {
            (true, true) => None,              // No children to advance
            (true, false) => Some(parent_idx), // Leaf node, nothing to do
            (false, false) => eclass_children
                .iter()
                .try_fold(parent_idx, |curr_idx, &child_id| {
                    self.next_choices(child_id, curr_idx + 1, false)
                }),
            (false, true) => self.advance_children(&eclass_children, parent_idx),
        }
    }

    /// Advance to the next combination by trying to advance rightmost child first.
    fn advance_children(&mut self, children: &[EClassId], parent_idx: usize) -> Option<usize> {
        // Try advancing each child from right to left
        (0..children.len()).rev().find_map(|advance_idx| {
            // Rebuild prefix (children before advance_idx)
            let prefix_idx = children[..advance_idx]
                .iter()
                .try_fold(parent_idx, |curr_idx, &child_id| {
                    self.next_choices(child_id, curr_idx + 1, false)
                })?;

            // Try to advance child at advance_idx
            let advanced_idx = self.next_choices(children[advance_idx], prefix_idx + 1, true)?;

            // Rebuild suffix (children after advance_idx)
            children[advance_idx + 1..]
                .iter()
                .try_fold(advanced_idx, |curr_idx, &child_id| {
                    self.next_choices(child_id, curr_idx + 1, false)
                })
        })
    }
}

impl<L: Label> Iterator for ChoiceIter<'_, L> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        // On first call, choices is empty, so advance=false finds the first tree.
        // On subsequent calls, choices contains the previous result, so advance=true
        // finds the next tree.
        let advance = !self.choices.is_empty();
        let root = self.egraph.root();

        self.next_choices(root, 0, advance)?;

        Some(self.choices.clone())
    }
}

/// Count the number of trees in an e-graph with the given revisit limit.
#[must_use]
pub fn count_trees<L: Label>(egraph: &EGraph<L>, max_revisits: usize) -> usize {
    let mut path = PathTracker::new(max_revisits);
    count_trees_rec(egraph, egraph.root(), &mut path)
}

fn count_trees_rec<L: Label>(egraph: &EGraph<L>, id: EClassId, path: &mut PathTracker) -> usize {
    // Cycle detection
    if !path.can_visit(id) {
        return 0;
    }

    path.enter(id);
    let count = egraph
        .class(id)
        .nodes()
        .iter()
        .map(|node| {
            node.children()
                .iter()
                .map(|child_id| {
                    if let ExprChildId::EClass(inner_id) = child_id {
                        count_trees_rec(egraph, *inner_id, path)
                    } else {
                        1
                    }
                })
                .product::<usize>() // product for children (and-choices)
        })
        .sum::<usize>(); // sum for nodes (or-choices)
    path.leave(id);
    count
}

/// Path tracker for cycle detection in the `EGraph`.
/// Tracks how many times each class has been visited on the current path
/// and allows configurable revisit limits.
#[derive(Debug, Clone)]
struct PathTracker {
    /// Visit counts for classes on the current path
    visits: HashMap<EClassId, usize>,
    /// Maximum number of times any node may be revisited (0 = no revisits allowed)
    max_revisits: usize,
}

impl PathTracker {
    fn new(max_revisits: usize) -> Self {
        PathTracker {
            visits: HashMap::new(),
            max_revisits,
        }
    }

    /// Check if visiting this OR node would exceed the revisit limit.
    /// Returns true if the visit is allowed.
    fn can_visit(&self, id: EClassId) -> bool {
        let count = self.visits.get(&id).copied().unwrap_or(0);
        count <= self.max_revisits
    }

    /// Mark an OR node as visited on the current path.
    fn enter(&mut self, id: EClassId) {
        *self.visits.entry(id).or_insert(0) += 1;
    }

    /// Unmark an OR node when leaving the current path.
    fn leave(&mut self, id: EClassId) {
        if let Some(count) = self.visits.get_mut(&id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.visits.remove(&id);
            }
        }
    }
}

/// Statistics from filtered extraction
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Total number of trees enumerated
    pub trees_enumerated: usize,
    /// Trees pruned by simple metric
    pub size_pruned: usize,
    /// Number of trees pruned by euler string filter
    pub euler_pruned: usize,
    /// Number of trees for which full distance was computed
    pub full_comparisons: usize,
}

impl Stats {
    pub(crate) fn size_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 1,
            euler_pruned: 0,
            full_comparisons: 0,
        }
    }

    pub(crate) fn euler_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            euler_pruned: 1,
            full_comparisons: 0,
        }
    }

    pub(crate) fn compared() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            euler_pruned: 0,
            full_comparisons: 1,
        }
    }
}

impl std::ops::Add for Stats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            trees_enumerated: self.trees_enumerated + rhs.trees_enumerated,
            size_pruned: self.size_pruned + rhs.size_pruned,
            euler_pruned: self.euler_pruned + rhs.euler_pruned,
            full_comparisons: self.full_comparisons + rhs.full_comparisons,
        }
    }
}

/// Find the tree in the e-graph with minimum Zhang-Shasha edit distance to the reference.
///
/// Uses parallel enumeration with pruning heuristics:
/// - Size difference lower bound
/// - Euler string distance lower bound
///
/// # Arguments
/// * `graph` - The e-graph to search
/// * `reference` - The target tree to match
/// * `costs` - Edit cost function
/// * `max_revisits` - Maximum allowed revisits for cycle handling
/// * `with_types` - Whether to include type annotations in comparison
///
/// # Returns
/// A tuple of (`best_result`, statistics) where `best_result` is `Some((tree, distance))`
/// if a tree was found.
#[must_use]
pub fn find_min_zs<L: Label, C: EditCosts<L>>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    costs: &C,
    max_revisits: usize,
    with_types: bool,
) -> (Option<(TreeNode<L>, usize)>, Stats) {
    let ref_tree = if with_types {
        reference
    } else {
        &reference.strip_types()
    };

    let ref_size = ref_tree.size();
    let ref_euler = EulerString::new(ref_tree);
    let ref_pp = PreprocessedTree::new(ref_tree);
    let running_best = AtomicUsize::new(usize::MAX);

    let (result, stats) = graph
        .choice_iter(max_revisits)
        .par_bridge()
        .progress_count(graph.count_trees(max_revisits) as u64)
        .map(|choices| {
            {
                let stripped_candidated =
                    graph.tree_from_choices(graph.root(), &choices, with_types);
                let best = running_best.load(Ordering::Relaxed);

                // Fast pruning: size difference is a lower bound on edit distance
                // (need at least |n1 - n2| insertions or deletions)
                if stripped_candidated.size().abs_diff(ref_size) > best {
                    return (None, Stats::size_pruned());
                }

                // Euler string heuristic: EDS(s(T1), s(T2)) ≤ 2 · EDT(T1, T2)
                // Therefore EDT ≥ EDS / 2, giving us a tighter lower bound
                if ref_euler.lower_bound(&stripped_candidated, costs) > best {
                    return (None, Stats::euler_pruned());
                }

                let distance = tree_distance_with_ref(&stripped_candidated, &ref_pp, costs);
                running_best.fetch_min(distance, Ordering::Relaxed);

                let tree = graph.tree_from_choices(graph.root(), &choices, true);
                (Some((tree, distance)), Stats::compared())
            }
        })
        .reduce(
            || (None, Stats::default()),
            |a, b| {
                let best = [a.0, b.0].into_iter().flatten().min_by_key(|v| v.1);
                (best, a.1 + b.1)
            },
        );

    (result, stats)
}

/// Find the tree in the e-graph with minimum structural difference to the reference.
///
/// # Arguments
/// * `graph` - The e-graph to search
/// * `reference` - The target tree to match
/// * `costs` - Edit cost function
/// * `max_revisits` - Maximum allowed revisits for cycle handling
/// * `with_types` - Whether to include type annotations in comparison
/// * `ignore_labels` - Whether to ignore label differences (structure only)
///
/// # Returns
/// `Some((tree, distance))` if a tree was found.
#[must_use]
pub fn find_min_struct<L: Label, C: EditCosts<L>>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    costs: &C,
    max_revisits: usize,
    with_types: bool,
    ignore_labels: bool,
) -> Option<(TreeNode<L>, usize)> {
    let ref_tree = if with_types {
        reference
    } else {
        &reference.strip_types()
    };
    graph
        .choice_iter(max_revisits)
        .par_bridge()
        .progress_count(graph.count_trees(max_revisits) as u64)
        .map(|choices| {
            let stripped_candidated = graph.tree_from_choices(graph.root(), &choices, with_types);
            let distance = structural_diff(ref_tree, &stripped_candidated, costs, ignore_labels);
            let tree = graph.tree_from_choices(graph.root(), &choices, true);
            (tree, distance)
        })
        .min_by_key(|(_, d)| *d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::graph::EClass;
    use crate::distance::ids::{ExprChildId, NatId, TypeChildId};
    use crate::distance::nodes::{ENode, NatNode};
    use crate::distance::zs::UnitCost;

    fn leaf<L: Label>(label: impl Into<L>) -> TreeNode<L> {
        TreeNode::leaf(label.into())
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::new(label, children)
    }

    fn eid(i: usize) -> ExprChildId {
        ExprChildId::EClass(EClassId::new(i))
    }

    fn dummy_ty() -> TypeChildId {
        TypeChildId::Nat(NatId::new(0))
    }

    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut nats = HashMap::new();
        nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        nats
    }

    fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
        classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (EClassId::new(i), c))
            .collect()
    }

    #[test]
    fn min_distance_exact_match() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("b".to_owned()), leaf("0".to_owned())],
                        ),
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("c".to_owned()), leaf("0".to_owned())],
                        ),
                    ],
                ),
                leaf("0".to_owned()),
            ],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
    }

    #[test]
    fn min_distance_chooses_best() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
        assert_eq!(result.0.label(), "typeOf");
        assert_eq!(result.0.children()[0].label(), "a");
    }

    #[test]
    fn min_distance_with_structure_choice() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("a".to_owned(), vec![eid(1)]),
                        ENode::new("a".to_owned(), vec![eid(1), eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![node(
                        "typeOf".to_owned(),
                        vec![leaf("b".to_owned()), leaf("0".to_owned())],
                    )],
                ),
                leaf("0".to_owned()),
            ],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
        assert_eq!(result.0.children().len(), 2);
        assert_eq!(result.0.children()[0].children().len(), 1);
    }

    #[test]
    fn min_distance_extract_fast_exact_match() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![
                node(
                    "a".to_owned(),
                    vec![
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("b".to_owned()), leaf("0".to_owned())],
                        ),
                        node(
                            "typeOf".to_owned(),
                            vec![leaf("c".to_owned()), leaf("0".to_owned())],
                        ),
                    ],
                ),
                leaf("0".to_owned()),
            ],
        );

        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();
        assert_eq!(result.1, 0);
    }

    #[test]
    fn min_distance_extract_fast_chooses_best() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();
        assert_eq!(result.1, 0);
        assert_eq!(result.0.label(), "typeOf");
        assert_eq!(result.0.children()[0].label(), "a");
    }

    #[test]
    fn min_distance_extract_filtered_prunes_bad_trees() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let (result, stats) = find_min_zs(&graph, &reference, &UnitCost, 0, true);

        assert_eq!(result.unwrap().1, 0);
        assert_eq!(stats.trees_enumerated, 2);
        assert_eq!(
            stats.size_pruned + stats.euler_pruned + stats.full_comparisons,
            stats.trees_enumerated
        );
    }

    #[test]
    fn choice_iter_enumerates_all_trees_diamond_cycle() {
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::new("b".to_owned(), vec![eid(3)])], dummy_ty()),
                EClass::new(vec![ENode::new("c".to_owned(), vec![eid(3)])], dummy_ty()),
                EClass::new(
                    vec![
                        ENode::new("rec".to_owned(), vec![eid(3)]),
                        ENode::leaf("d".to_owned()),
                    ],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        assert_eq!(graph.choice_iter(1).count(), 4);
        assert_eq!(graph.count_trees(1), graph.choice_iter(1).count());

        let trees = graph
            .choice_iter(1)
            .map(|c| graph.tree_from_choices(graph.root(), &c, false).to_string())
            .collect::<Vec<_>>();
        assert!(trees.contains(&"(a (b d) (c d))".to_owned()));
        assert!(trees.contains(&"(a (b d) (c (rec d)))".to_owned()));
        assert!(trees.contains(&"(a (b (rec d)) (c d))".to_owned()));
        assert!(trees.contains(&"(a (b (rec d)) (c (rec d)))".to_owned()));

        assert_eq!(graph.choice_iter(0).count(), 1);
        assert_eq!(graph.count_trees(0), graph.choice_iter(0).count());
    }
}
