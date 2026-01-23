//! `EGraph` Extension for Zhang-Shasha Tree Edit Distance
//!
//! Finds the solution tree in a bounded `EGraph` with minimum edit distance
//! to a target tree. Assumes bounded maximum number of nodes in an `EClass` (N) and bounded depth (d).
//!
//! With strict alternation (`EClass` -> `ENode` -> `EClass` ->...),
//! complexity is O(N^(d/2) * |T|^2) for single-path graphs

use std::hash::Hash;

use hashbrown::HashMap;

use super::TreeNode;
use super::tree::{EditCosts, UnitCost, tree_distance};

/// Trait for node labels
pub trait Label: Clone + Eq + Hash + std::fmt::Debug {}

// Blanket implementation for any type that satisfies the bounds
impl<T: Clone + Eq + Hash + std::fmt::Debug> Label for T {}

/// `ENode` must take all children
/// Children are indices into the `EGraph` array
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode<L: Label> {
    pub label: L,
    pub children: Vec<Id>, // indices into EClass array
}

/// `EClass`: choose exactly one child (`ENode`)
/// Children are `ENode` instances directly
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EClass<L: Label> {
    pub label: L,
    pub children: Vec<ENode<L>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EGraph<L: Label> {
    classes: Vec<EClass<L>>,
    root: Id,
}

impl<L: Label> EGraph<L> {
    #[must_use]
    pub fn new(classes: Vec<EClass<L>>, root: Id) -> Self {
        Self { classes, root }
    }

    #[must_use]
    pub fn class(&self, id: Id) -> &EClass<L> {
        &self.classes[id.0]
    }

    /// Enumerate all possible trees extractable from an `EGraph` starting at a given class,
    /// respecting the cycle revisit limit.
    fn enumerate_trees_from_class(
        &self,
        class_id: Id,
        path: &mut PathTracker,
    ) -> Option<Vec<TreeNode<L>>> {
        // Check if we can visit this class
        if !path.can_visit(class_id) {
            return None;
        }

        path.enter(class_id);

        let class = self.class(class_id);
        let mut all_trees = Vec::new();

        // OR choice: try each node in the class
        for node in &class.children {
            // AND combination: must take all children
            // Recursively enumerate trees for each child class
            let child_tree_lists = node
                .children
                .iter()
                .map(|&child_id| self.enumerate_trees_from_class(child_id, path))
                .collect::<Option<Vec<_>>>()?;

            // Cartesian product of all child tree combinations
            let child_combinations = cartesian_product(&child_tree_lists);

            // Build a tree for each combination
            for children in child_combinations {
                let tree = TreeNode::with_children(node.label.clone(), children);
                all_trees.push(tree);
            }
        }

        path.leave(class_id);

        Some(all_trees)
    }

    /// Enumerate all possible trees extractable from an `EGraph`,
    /// respecting the maximum cycle revisit limit.
    #[must_use]
    pub fn enumerate_trees(&self, max_revisits: usize) -> Vec<TreeNode<L>> {
        let mut path = PathTracker::new(max_revisits);
        self.enumerate_trees_from_class(self.root, &mut path)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(usize);

impl Id {
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl From<Id> for usize {
    fn from(value: Id) -> Self {
        value.0
    }
}

/// Path tracker for cycle detection in the `EGraph`.
/// Tracks how many times each class has been visited on the current path
/// and allows configurable revisit limits.
///
/// Only classes are tracked because nodes and classes strictly alternate.
/// Cycles are detected at classes since that's where paths can reconverge.
#[derive(Debug, Clone)]
struct PathTracker {
    /// Visit counts for classes on the current path
    visits: HashMap<Id, usize>,
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
    fn can_visit(&self, id: Id) -> bool {
        let count = self.visits.get(&id).copied().unwrap_or(0);
        count <= self.max_revisits
    }

    /// Mark an OR node as visited on the current path.
    fn enter(&mut self, id: Id) {
        *self.visits.entry(id).or_insert(0) += 1;
    }

    /// Unmark an OR node when leaving the current path.
    fn leave(&mut self, id: Id) {
        if let Some(count) = self.visits.get_mut(&id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.visits.remove(&id);
            }
        }
    }
}

/// Result of finding the minimum edit distance solution tree
#[derive(Debug, Clone)]
pub struct MinEditResult<L: Clone + Eq> {
    pub tree: TreeNode<L>,
    pub distance: usize,
}

/// Compute cartesian product of vectors
fn cartesian_product<T: Clone>(lists: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut result = vec![vec![]];

    for list in lists {
        let mut new_result = Vec::new();
        for existing in &result {
            for item in list {
                let mut new_vec = existing.clone();
                new_vec.push(item.clone());
                new_result.push(new_vec);
            }
        }
        result = new_result;
    }

    result
}

/// Find the tree in the `EGraph` with minimum edit distance to the reference tree.
///
/// This uses enumeration: it extracts all possible trees from the graph (bounded by
/// `max_revisits` for cycles) and computes the full Zhang-Shasha distance for each.
///
/// Returns None if no valid trees can be extracted from the graph.
pub fn min_distance_extract<L: Label, C: EditCosts<L>>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    max_revisits: usize,
    costs: &C,
) -> Option<MinEditResult<L>> {
    graph
        .enumerate_trees(max_revisits)
        .into_iter()
        .map(|tree| {
            let distance = tree_distance(&tree, reference, costs);
            MinEditResult { tree, distance }
        })
        .min_by_key(|result| result.distance)
}

/// Convenience function using unit costs
pub fn min_distance_extract_unit<L: Label>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    max_revisits: usize,
) -> Option<MinEditResult<L>> {
    min_distance_extract(graph, reference, max_revisits, &UnitCost)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf<L: Label>(label: L) -> TreeNode<L> {
        TreeNode::new(label)
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::with_children(label, children)
    }

    /// Helper to build a simple graph with one class containing one node
    fn single_node_graph(label: &str) -> EGraph<&str> {
        EGraph::new(
            vec![EClass {
                label,
                children: vec![ENode {
                    label,
                    children: vec![],
                }],
            }],
            Id::new(0),
        )
    }

    #[test]
    fn enumerate_single_leaf() {
        let graph = single_node_graph("a");
        let trees = graph.enumerate_trees(0);

        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), &"a");
        assert!(trees[0].is_leaf());
    }

    #[test]
    fn enumerate_with_or_choice() {
        // Graph with one class containing two node choices
        let graph = EGraph::new(
            vec![EClass {
                label: "root",
                children: vec![
                    ENode {
                        label: "a",
                        children: vec![],
                    },
                    ENode {
                        label: "b",
                        children: vec![],
                    },
                ],
            }],
            Id::new(0),
        );

        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 2);

        let labels: Vec<_> = trees.iter().map(|t| t.label()).collect();
        assert!(labels.contains(&&"a"));
        assert!(labels.contains(&&"b"));
    }

    #[test]
    fn enumerate_with_and_children() {
        // Graph: root class -> node with two child classes (each has one leaf)
        // Class 0: root, has node "a" pointing to classes 1 and 2
        // Class 1: leaf "b"
        // Class 2: leaf "c"
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![ENode {
                        label: "a",
                        children: vec![Id(1), Id(2)],
                    }],
                },
                EClass {
                    label: "b",
                    children: vec![ENode {
                        label: "b",
                        children: vec![],
                    }],
                },
                EClass {
                    label: "c",
                    children: vec![ENode {
                        label: "c",
                        children: vec![],
                    }],
                },
            ],
            Id::new(0),
        );

        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), &"a");
        assert_eq!(trees[0].children().len(), 2);
        assert_eq!(trees[0].children()[0].label(), &"b");
        assert_eq!(trees[0].children()[1].label(), &"c");
    }

    #[test]
    fn enumerate_with_cycle_no_revisits() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            vec![EClass {
                label: "a",
                children: vec![
                    ENode {
                        label: "a",
                        children: vec![Id(0)], // points back to self
                    },
                    ENode {
                        label: "leaf",
                        children: vec![],
                    },
                ],
            }],
            Id::new(0),
        );

        // With 0 revisits, we can only take the leaf option
        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), &"leaf");
    }

    #[test]
    fn enumerate_with_cycle_one_revisit() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            vec![EClass {
                label: "a",
                children: vec![
                    ENode {
                        label: "rec",
                        children: vec![Id(0)], // points back to self
                    },
                    ENode {
                        label: "leaf",
                        children: vec![],
                    },
                ],
            }],
            Id::new(0),
        );

        // With 1 revisit, we can go one level deep
        let trees = graph.enumerate_trees(1);

        // Should have: "leaf", "rec(leaf)", "rec(rec(leaf))"
        // Actually: at depth 0 we have 2 choices, at depth 1 we have 2 choices...
        // Let's verify we get more trees than with 0 revisits
        assert!(trees.len() > 1);

        // Check that we have the recursive structure
        let has_recursive = trees
            .iter()
            .any(|t| t.label() == &"rec" && !t.children().is_empty());
        assert!(has_recursive);
    }

    #[test]
    fn min_distance_exact_match() {
        // Graph contains the exact reference tree
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "a",
                    children: vec![ENode {
                        label: "a",
                        children: vec![Id(1), Id(2)],
                    }],
                },
                EClass {
                    label: "b",
                    children: vec![ENode {
                        label: "b",
                        children: vec![],
                    }],
                },
                EClass {
                    label: "c",
                    children: vec![ENode {
                        label: "c",
                        children: vec![],
                    }],
                },
            ],
            Id::new(0),
        );

        let reference = node("a", vec![leaf("b"), leaf("c")]);
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
    }

    #[test]
    fn min_distance_chooses_best() {
        // Graph with OR choice: "a" or "x"
        // Reference is "a", so should choose "a" with distance 0
        let graph = EGraph::new(
            vec![EClass {
                label: "root",
                children: vec![
                    ENode {
                        label: "a",
                        children: vec![],
                    },
                    ENode {
                        label: "x",
                        children: vec![],
                    },
                ],
            }],
            Id::new(0),
        );

        let reference = leaf("a");
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
        assert_eq!(result.tree.label(), &"a");
    }

    #[test]
    fn min_distance_with_structure_choice() {
        // Graph offers two structures:
        // Option 1: a(b)
        // Option 2: a(b, c)
        // Reference: a(b)
        // Should choose option 1 with distance 0
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "a",
                    children: vec![
                        ENode {
                            label: "a",
                            children: vec![Id(1)], // a(b)
                        },
                        ENode {
                            label: "a",
                            children: vec![Id(1), Id(2)], // a(b, c)
                        },
                    ],
                },
                EClass {
                    label: "b",
                    children: vec![ENode {
                        label: "b",
                        children: vec![],
                    }],
                },
                EClass {
                    label: "c",
                    children: vec![ENode {
                        label: "c",
                        children: vec![],
                    }],
                },
            ],
            Id::new(0),
        );

        let reference = node("a", vec![leaf("b")]);
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
        assert_eq!(result.tree.children().len(), 1);
    }
}
