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
    label: L,
    children: Vec<Id>, // indices into EClass array
}

impl<L: Label> ENode<L> {
    pub fn new_leaf(label: L) -> Self {
        Self {
            label,
            children: Vec::new(),
        }
    }

    pub fn new_with_children(label: L, children: Vec<Id>) -> Self {
        Self { label, children }
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    pub fn children(&self) -> &[Id] {
        &self.children
    }
}

/// `EClass`: choose exactly one child (`ENode`)
/// Children are `ENode` instances directly
/// Must have at least one child
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EClass<L: Label> {
    label: L,
    children: Vec<ENode<L>>,
}

impl<L: Label> EClass<L> {
    pub fn new(label: L, children: Vec<ENode<L>>) -> Self {
        Self { label, children }
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    pub fn children(&self) -> &[ENode<L>] {
        &self.children
    }
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
    ///
    /// Returns the tree and the index of the last choice used, or None if no more trees exist.
    /// The `choices` vector is modified to record/follow choices at each `EClass`.
    fn get_next_tree(
        &self,
        id: Id,
        choice_idx: usize,
        choices: &mut Vec<usize>,
        path: &mut PathTracker,
    ) -> Option<(TreeNode<L>, usize)> {
        // Cycle detection
        if !path.can_visit(id) {
            return None;
        }

        let class = self.class(id);
        let choice = choices.get(choice_idx).copied().unwrap_or_else(|| {
            choices.push(0);
            0
        });

        path.enter(id);
        // Try choices starting from `choice`, looking for a valid one
        for (node_idx, node) in class.children.iter().enumerate().skip(choice) {
            // Set the choice_idx for future choices
            // Useless write if the choice is correct
            choices[choice_idx] = node_idx;

            let result = node.children.iter().try_fold(
                (Vec::new(), choice_idx),
                |(mut children, curr_idx), child_id| {
                    let (child_tree, last_idx) =
                        self.get_next_tree(*child_id, curr_idx + 1, choices, path)?;
                    children.push(child_tree);
                    Some((children, last_idx))
                },
            );

            if let Some((children, curr_idx)) = result {
                path.leave(id);
                let tree = TreeNode::with_children(node.label.clone(), children);
                return Some((tree, curr_idx));
            }

            // This node's children failed, try next node in this class
            // Reset choices for children (truncate to current position + 1)
            choices.truncate(choice_idx + 1);
        }

        path.leave(id);
        // No valid choice found at this level, need to backtrack
        None
    }

    pub fn tree_from_choices(
        &self,
        id: Id,
        choice_idx: usize,
        choices: &mut Vec<usize>,
    ) -> TreeNode<L> {
        fn rec<L: Label>(
            graph: &EGraph<L>,
            id: Id,
            choice_idx: usize,
            choices: &mut Vec<usize>,
        ) -> (TreeNode<L>, usize) {
            let class = graph.class(id);
            let choice = choices[choice_idx];
            let node = &class.children[choice];

            let (children, curr_idx) = node.children.iter().fold(
                (Vec::new(), choice_idx),
                |(mut children, curr_idx), child_id| {
                    let (child_tree, last_idx) = rec(graph, *child_id, curr_idx + 1, choices);
                    children.push(child_tree);
                    (children, last_idx)
                },
            );

            let tree = TreeNode::with_children(node.label.clone(), children);
            (tree, curr_idx)
        }
        rec(self, id, choice_idx, choices).0
    }

    #[must_use]
    pub fn enumerate_trees(&self, max_revisits: usize) -> Vec<TreeNode<L>> {
        TreeIter::new(self, max_revisits).collect()
    }

    #[must_use]
    pub fn count_trees(&self, max_revisits: usize) -> usize {
        TreeIter::new(self, max_revisits).count()
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

pub struct TreeIter<'a, L: Label> {
    choices: Vec<usize>,
    path: PathTracker,
    egraph: &'a EGraph<L>,
}

impl<'a, L: Label> TreeIter<'a, L> {
    pub fn new(egraph: &'a EGraph<L>, max_revisits: usize) -> Self {
        Self {
            choices: Vec::new(),
            path: PathTracker::new(max_revisits),
            egraph,
        }
    }
}

impl<L: Label> Iterator for TreeIter<'_, L> {
    type Item = TreeNode<L>;

    fn next(&mut self) -> Option<Self::Item> {
        let (tree, _) =
            self.egraph
                .get_next_tree(self.egraph.root, 0, &mut self.choices, &mut self.path)?;
        if let Some(last) = self.choices.last_mut() {
            *last += 1;
        }
        Some(tree)
    }
}

/// Path tracker for cycle detection in the `EGraph`.
/// Tracks how many times each class has been visited on the current path
/// and allows configurable revisit limits.
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

/// Find the tree in the `EGraph` with minimum edit distance to the reference tree.
///
/// This uses dynamic enumeration: it extracts all possible trees from the graph (bounded by
/// `max_revisits` for cycles) and computes the full Zhang-Shasha distance for each.
///
/// Returns None if no valid trees can be extracted from the graph.
pub fn min_distance_extract<L: Label, C: EditCosts<L>>(
    graph: &EGraph<L>,
    reference: &TreeNode<L>,
    max_revisits: usize,
    costs: &C,
) -> Option<MinEditResult<L>> {
    TreeIter::new(graph, max_revisits)
        .map(|tree| {
            let distance = tree_distance(&tree, reference, costs);
            MinEditResult { tree, distance }
        })
        .min_by_key(|result| result.distance)
}

/// See `min_distance_extract` but with unit costs
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

    #[test]
    fn tree_from_choices_single_leaf() {
        let graph = single_node_graph("a");
        let mut choices = vec![0];

        let tree = graph.tree_from_choices(Id::new(0), 0, &mut choices);

        assert_eq!(tree.label(), &"a");
        assert!(tree.is_leaf());
    }

    #[test]
    fn tree_from_choices_or_choice_first() {
        // Graph with two OR choices
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

        let mut choices = vec![0];
        let tree = graph.tree_from_choices(Id::new(0), 0, &mut choices);

        assert_eq!(tree.label(), &"a");
    }

    #[test]
    fn tree_from_choices_or_choice_second() {
        // Graph with two OR choices
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

        let mut choices = vec![1];
        let tree = graph.tree_from_choices(Id::new(0), 0, &mut choices);

        assert_eq!(tree.label(), &"b");
    }

    #[test]
    fn tree_from_choices_with_and_children() {
        // Graph: root -> node with two child classes
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

        let mut choices = vec![0, 0, 0];
        let tree = graph.tree_from_choices(Id::new(0), 0, &mut choices);

        assert_eq!(tree.label(), &"a");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), &"b");
        assert_eq!(tree.children()[1].label(), &"c");
    }

    #[test]
    fn tree_from_choices_nested_or_choices() {
        // Graph with nested OR choices:
        // Class 0: root with two node options
        //   Node "x" -> points to Class 1
        //   Node "y" -> points to Class 1
        // Class 1: two leaf options "a" or "b"
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![
                        ENode {
                            label: "x",
                            children: vec![Id(1)],
                        },
                        ENode {
                            label: "y",
                            children: vec![Id(1)],
                        },
                    ],
                },
                EClass {
                    label: "leaf",
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
                },
            ],
            Id::new(0),
        );

        // Test x(a)
        let mut choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(Id::new(0), 0, &mut choices1);
        assert_eq!(tree1.label(), &"x");
        assert_eq!(tree1.children()[0].label(), &"a");

        // Test x(b)
        let mut choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(Id::new(0), 0, &mut choices2);
        assert_eq!(tree2.label(), &"x");
        assert_eq!(tree2.children()[0].label(), &"b");

        // Test y(a)
        let mut choices3 = vec![1, 0];
        let tree3 = graph.tree_from_choices(Id::new(0), 0, &mut choices3);
        assert_eq!(tree3.label(), &"y");
        assert_eq!(tree3.children()[0].label(), &"a");

        // Test y(b)
        let mut choices4 = vec![1, 1];
        let tree4 = graph.tree_from_choices(Id::new(0), 0, &mut choices4);
        assert_eq!(tree4.label(), &"y");
        assert_eq!(tree4.children()[0].label(), &"b");
    }

    #[test]
    fn tree_from_choices_multiple_and_children_with_or_choices() {
        // Graph with multiple AND children each having OR choices:
        // Class 0: root -> "p" with children [Class 1, Class 2]
        // Class 1: "a" or "b"
        // Class 2: "x" or "y"
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![ENode {
                        label: "p",
                        children: vec![Id(1), Id(2)],
                    }],
                },
                EClass {
                    label: "left",
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
                },
                EClass {
                    label: "right",
                    children: vec![
                        ENode {
                            label: "x",
                            children: vec![],
                        },
                        ENode {
                            label: "y",
                            children: vec![],
                        },
                    ],
                },
            ],
            Id::new(0),
        );

        // Test p(a, x)
        let mut choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(Id::new(0), 0, &mut choices1);
        assert_eq!(tree1.label(), &"p");
        assert_eq!(tree1.children()[0].label(), &"a");
        assert_eq!(tree1.children()[1].label(), &"x");

        // Test p(a, y)
        let mut choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(Id::new(0), 0, &mut choices2);
        assert_eq!(tree2.label(), &"p");
        assert_eq!(tree2.children()[0].label(), &"a");
        assert_eq!(tree2.children()[1].label(), &"y");

        // Test p(b, x)
        let mut choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(Id::new(0), 0, &mut choices3);
        assert_eq!(tree3.label(), &"p");
        assert_eq!(tree3.children()[0].label(), &"b");
        assert_eq!(tree3.children()[1].label(), &"x");

        // Test p(b, y)
        let mut choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(Id::new(0), 0, &mut choices4);
        assert_eq!(tree4.label(), &"p");
        assert_eq!(tree4.children()[0].label(), &"b");
        assert_eq!(tree4.children()[1].label(), &"y");
    }

    #[test]
    fn tree_from_choices_deep_nesting() {
        // Deep tree with choices at multiple levels
        // Class 0: root -> "a" with child [Class 1]
        // Class 1: "b1" or "b2", both with child [Class 2]
        // Class 2: "c1" or "c2"
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![ENode {
                        label: "a",
                        children: vec![Id(1)],
                    }],
                },
                EClass {
                    label: "middle",
                    children: vec![
                        ENode {
                            label: "b1",
                            children: vec![Id(2)],
                        },
                        ENode {
                            label: "b2",
                            children: vec![Id(2)],
                        },
                    ],
                },
                EClass {
                    label: "bottom",
                    children: vec![
                        ENode {
                            label: "c1",
                            children: vec![],
                        },
                        ENode {
                            label: "c2",
                            children: vec![],
                        },
                    ],
                },
            ],
            Id::new(0),
        );

        // Test a(b1(c1))
        let mut choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(Id::new(0), 0, &mut choices1);
        assert_eq!(tree1.label(), &"a");
        assert_eq!(tree1.children()[0].label(), &"b1");
        assert_eq!(tree1.children()[0].children()[0].label(), &"c1");

        // Test a(b1(c2))
        let mut choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(Id::new(0), 0, &mut choices2);
        assert_eq!(tree2.label(), &"a");
        assert_eq!(tree2.children()[0].label(), &"b1");
        assert_eq!(tree2.children()[0].children()[0].label(), &"c2");

        // Test a(b2(c1))
        let mut choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(Id::new(0), 0, &mut choices3);
        assert_eq!(tree3.label(), &"a");
        assert_eq!(tree3.children()[0].label(), &"b2");
        assert_eq!(tree3.children()[0].children()[0].label(), &"c1");

        // Test a(b2(c2))
        let mut choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(Id::new(0), 0, &mut choices4);
        assert_eq!(tree4.label(), &"a");
        assert_eq!(tree4.children()[0].label(), &"b2");
        assert_eq!(tree4.children()[0].children()[0].label(), &"c2");
    }

    #[test]
    fn tree_from_choices_three_and_children() {
        // Test with three AND children
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![ENode {
                        label: "f",
                        children: vec![Id(1), Id(2), Id(3)],
                    }],
                },
                EClass {
                    label: "c1",
                    children: vec![ENode {
                        label: "a",
                        children: vec![],
                    }],
                },
                EClass {
                    label: "c2",
                    children: vec![ENode {
                        label: "b",
                        children: vec![],
                    }],
                },
                EClass {
                    label: "c3",
                    children: vec![ENode {
                        label: "c",
                        children: vec![],
                    }],
                },
            ],
            Id::new(0),
        );

        let mut choices = vec![0, 0, 0, 0];
        let tree = graph.tree_from_choices(Id::new(0), 0, &mut choices);

        assert_eq!(tree.label(), &"f");
        assert_eq!(tree.children().len(), 3);
        assert_eq!(tree.children()[0].label(), &"a");
        assert_eq!(tree.children()[1].label(), &"b");
        assert_eq!(tree.children()[2].label(), &"c");
    }

    #[test]
    fn tree_from_choices_matches_enumeration() {
        // Helper to check if two trees are structurally equal
        fn trees_equal<L: Label>(a: &TreeNode<L>, b: &TreeNode<L>) -> bool {
            if a.label() != b.label() || a.children().len() != b.children().len() {
                return false;
            }
            a.children()
                .iter()
                .zip(b.children().iter())
                .all(|(x, y)| trees_equal(x, y))
        }

        // Verify that tree_from_choices produces the same trees as enumeration
        let graph = EGraph::new(
            vec![
                EClass {
                    label: "root",
                    children: vec![
                        ENode {
                            label: "x",
                            children: vec![Id(1)],
                        },
                        ENode {
                            label: "y",
                            children: vec![],
                        },
                    ],
                },
                EClass {
                    label: "leaf",
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
                },
            ],
            Id::new(0),
        );

        let enumerated = graph.enumerate_trees(0);

        // Should produce: x(a), x(b), y
        assert_eq!(enumerated.len(), 3);

        // Reconstruct using tree_from_choices
        let mut choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(Id::new(0), 0, &mut choices1);
        assert!(trees_equal(&tree1, &enumerated[0]));

        let mut choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(Id::new(0), 0, &mut choices2);
        assert!(trees_equal(&tree2, &enumerated[1]));

        let mut choices3 = vec![1];
        let tree3 = graph.tree_from_choices(Id::new(0), 0, &mut choices3);
        assert!(trees_equal(&tree3, &enumerated[2]));
    }
}
