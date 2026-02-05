//! Minimum distance search for E-Graphs.
//!
//! This module provides functions for finding the tree with minimum edit distance to a reference.

use std::sync::atomic::{AtomicUsize, Ordering};

use indicatif::ParallelProgressIterator;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::distance::sampling::find_lambda_for_target_size;
use crate::distance::{FixpointSampler, FixpointSamplerConfig, Sampler};

use super::graph::EGraph;
use super::nodes::Label;
use super::str::EulerString;
use super::structural::structural_diff;
use super::tree::TreeNode;
use super::zs::{EditCosts, PreprocessedTree, tree_distance_with_ref};

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
pub fn find_min_exhaustive_zs<L: Label, C: EditCosts<L>>(
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

/// Find the tree in the e-graph with minimum Zhang-Shasha edit distance to the reference.
///
/// Uses fixpoint-based sampling instead of exhaustive enumeration. This is more efficient
/// for large e-graphs where exhaustive search is infeasible. Samples are drawn according to
/// a Boltzmann distribution that favors trees of a target weight.
///
/// Uses the same parallel pruning heuristics as exhaustive search:
/// - Size difference lower bound
/// - Euler string distance lower bound
///
/// # Arguments
/// * `graph` - The e-graph to search
/// * `reference` - The target tree to match
/// * `costs` - Edit cost function
/// * `with_types` - Whether to include type annotations in comparison
/// * `n_samples` - Number of trees to sample from the e-graph
/// * `target_weight` - Target tree size for the Boltzmann distribution (controls sampling bias)
/// * `seed` - Random seed for reproducible sampling
///
/// # Returns
/// A tuple of (`best_result`, statistics) where `best_result` is `Some((tree, distance))`
/// if a tree was found.
///
/// # Panics
///
/// Panics if no sampler can be built
///
/// # Note
/// The critical lambda parameter is automatically computed to target trees of the specified
#[must_use]
pub fn find_min_sampling_zs<L: Label, C: EditCosts<L>>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    costs: &C,
    with_types: bool,
    n_samples: usize,
    target_weight: usize,
    seed: u64,
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

    let mut rng = StdRng::seed_from_u64(seed);
    let config = FixpointSamplerConfig::builder().build();
    let (lambda, expected_size) =
        find_lambda_for_target_size(graph, target_weight, &config, with_types, &mut rng).unwrap();
    eprintln!("LAMBDA IS {lambda}");
    eprintln!("EXPECTED SIZE IS {expected_size}");
    let (result, stats) = FixpointSampler::new(graph, lambda, &config, rng)
        .unwrap()
        .into_sample_iter(with_types)
        .take(n_samples)
        .par_bridge()
        .progress_count(n_samples as u64)
        .map(|candidate| {
            {
                let candidate_tree = if with_types {
                    &candidate
                } else {
                    &candidate.strip_types()
                };
                let best = running_best.load(Ordering::Relaxed);

                // Fast pruning: size difference is a lower bound on edit distance
                // (need at least |n1 - n2| insertions or deletions)
                if candidate_tree.size().abs_diff(ref_size) > best {
                    return (None, Stats::size_pruned());
                }

                // Euler string heuristic: EDS(s(T1), s(T2)) ≤ 2 · EDT(T1, T2)
                // Therefore EDT ≥ EDS / 2, giving us a tighter lower bound
                if ref_euler.lower_bound(candidate_tree, costs) > best {
                    return (None, Stats::euler_pruned());
                }

                let distance = tree_distance_with_ref(candidate_tree, &ref_pp, costs);
                running_best.fetch_min(distance, Ordering::Relaxed);

                (Some((candidate, distance)), Stats::compared())
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
    use hashbrown::HashMap;

    use super::*;
    use crate::distance::graph::EClass;
    use crate::distance::ids::{EClassId, ExprChildId, NatId, TypeChildId};
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
        let result = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true)
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
        let result = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true)
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
        let result = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true)
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

        let result = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true)
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

        let result = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true)
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

        let (result, stats) = find_min_exhaustive_zs(&graph, &reference, &UnitCost, 0, true);

        assert_eq!(result.unwrap().1, 0);
        assert_eq!(stats.trees_enumerated, 2);
        assert_eq!(
            stats.size_pruned + stats.euler_pruned + stats.full_comparisons,
            stats.trees_enumerated
        );
    }
}
