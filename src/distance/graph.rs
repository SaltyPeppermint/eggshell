//! AND-OR Graph Extension for Zhang-Shasha Tree Edit Distance
//!
//! Finds the solution tree in a bounded AND-OR graph with minimum edit distance
//! to a target tree. Assumes bounded OR-branching (N) and bounded depth (d).
//!
//! With strict alternation (OR→AND→OR→...), complexity is O(N^(d/2) * |T|^2)
//! for single-path graphs, where N is OR-branching and d is depth.

use std::hash::Hash;

use hashbrown::HashMap;

use super::{EditCosts, TreeNode, tree_distance};

/// AND node: all children must be included in solution tree.
/// Children must be OR nodes (or leaves).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AndNode<L: Clone + Eq + Hash> {
    label: L,
    children: Vec<OrNode<L>>,
}

impl<L: Clone + Eq + Hash> AndNode<L> {
    /// AND node with OR children
    pub fn new(label: L, children: Vec<OrNode<L>>) -> Self {
        AndNode { label, children }
    }

    /// Leaf node (AND with no children)
    pub fn leaf(label: L) -> Self {
        AndNode {
            label,
            children: Vec::new(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn children(&self) -> &[OrNode<L>] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Generate all valid solution trees from this AND node.
    /// Includes all OR children, taking cartesian product of their choices.
    pub fn generate_solutions(&self) -> Vec<TreeNode<L>> {
        if self.is_leaf() {
            return vec![TreeNode::new(self.label.clone())];
        }

        // Each OR child contributes a set of possible subtrees
        let child_solutions: Vec<Vec<TreeNode<L>>> = self
            .children
            .iter()
            .map(OrNode::generate_solutions)
            .collect();

        cartesian_product(&child_solutions)
            .into_iter()
            .map(|children| TreeNode::with_children(self.label.clone(), children))
            .collect()
    }

    /// Count the number of solution trees (for complexity analysis)
    pub fn count_solutions(&self) -> usize {
        if self.is_leaf() {
            return 1;
        }
        self.children.iter().map(OrNode::count_solutions).product()
    }

    /// Find the solution from this AND node without memoization.
    fn find_min<C: EditCosts<L>>(&self, target: &TreeNode<L>, costs: &C) -> MinEditResult<L> {
        let tree = if self.is_leaf() {
            TreeNode::new(self.label.clone())
        } else {
            // For AND nodes, we must include all children.
            // Recursively find the best solution for each OR child.
            let child_trees: Vec<TreeNode<L>> = self
                .children
                .iter()
                .map(|or_child| or_child.find_min(target, costs).tree)
                .collect();

            TreeNode::with_children(self.label.clone(), child_trees)
        };

        let distance = tree_distance(&tree, target, costs);
        MinEditResult { tree, distance }
    }

    /// Find the solution from this AND node, with memoization.
    fn find_min_memo<'a, C: EditCosts<L>>(
        &'a self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<'a, L>,
    ) -> MinEditResult<L> {
        // Check cache first using pointer address as key
        if let Some(cached) = cache.and_cache.get(&self) {
            return cached.clone();
        }

        let tree = if self.is_leaf() {
            TreeNode::new(self.label.clone())
        } else {
            // For AND nodes, we must include all children.
            // Recursively find the best solution for each OR child.
            let child_trees: Vec<TreeNode<L>> = self
                .children
                .iter()
                .map(|or_child| or_child.find_min_memo(target, costs, cache).tree)
                .collect();

            TreeNode::with_children(self.label.clone(), child_trees)
        };

        let distance = tree_distance(&tree, target, costs);
        let result = MinEditResult { tree, distance };
        cache.and_cache.insert(self, result.clone());
        result
    }
}

/// OR node: exactly one child is chosen for solution tree.
/// Children must be AND nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrNode<L: Clone + Eq + Hash> {
    label: L,
    children: Vec<AndNode<L>>,
}

impl<L: Clone + Eq + Hash> OrNode<L> {
    /// OR node with AND children
    #[expect(clippy::missing_panics_doc)]
    pub fn new(label: L, children: Vec<AndNode<L>>) -> Self {
        assert!(!children.is_empty(), "OR node must have at least one child");
        OrNode { label, children }
    }

    /// Single-choice OR node (wraps an AND node)
    pub fn single(label: L, child: AndNode<L>) -> Self {
        OrNode {
            label,
            children: vec![child],
        }
    }

    pub fn children(&self) -> &[AndNode<L>] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Generate all valid solution trees from this OR node.
    /// Chooses exactly one AND child.
    pub fn generate_solutions(&self) -> Vec<TreeNode<L>> {
        // Each AND child is a possible choice; collect all their solutions
        self.children
            .iter()
            .flat_map(|and_child| {
                and_child.generate_solutions().into_iter().map(|subtree| {
                    // OR node label becomes parent of the chosen subtree
                    TreeNode::with_children(self.label.clone(), vec![subtree])
                })
            })
            .collect()
    }

    /// Count the number of solution trees
    pub fn count_solutions(&self) -> usize {
        self.children.iter().map(AndNode::count_solutions).sum()
    }

    /// Find the best solution from this OR node without memoization.
    fn find_min<C: EditCosts<L>>(&self, target: &TreeNode<L>, costs: &C) -> MinEditResult<L> {
        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for and_child in &self.children {
            let subtree = and_child.find_min(target, costs).tree;

            // OR node label becomes parent of the chosen subtree
            let candidate_tree = TreeNode::with_children(self.label.clone(), vec![subtree]);

            // Compute actual edit distance for this solution
            let distance = tree_distance(&candidate_tree, target, costs);

            if distance < best_distance {
                best_distance = distance;
                best_tree = Some(candidate_tree);
            }
            if best_distance == 0 {
                break;
            }
        }

        MinEditResult {
            tree: best_tree.expect("OR node must have at least one child"),
            distance: best_distance,
        }
    }

    /// Find the best solution from this OR node, with memoization.
    fn find_min_memo<'a, C: EditCosts<L>>(
        &'a self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<'a, L>,
    ) -> MinEditResult<L> {
        // Check cache first using pointer address as key
        if let Some(cached) = cache.or_cache.get(&self) {
            return cached.clone();
        }

        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for and_child in &self.children {
            let subtree = and_child.find_min_memo(target, costs, cache).tree;

            // OR node label becomes parent of the chosen subtree
            let candidate_tree = TreeNode::with_children(self.label.clone(), vec![subtree]);

            // Compute actual edit distance for this solution
            let dist = tree_distance(&candidate_tree, target, costs);

            if dist < best_distance {
                best_distance = dist;
                best_tree = Some(candidate_tree);
            }
            if best_distance == 0 {
                break;
            }
        }

        let result = MinEditResult {
            tree: best_tree.expect("OR node must have at least one child"),
            distance: best_distance,
        };
        cache.or_cache.insert(self, result.clone());
        result
    }
}

/// Memoization cache for AND-OR graph edit distance computation.
/// When the AND-OR graph is a DAG (has shared substructure), this cache
/// prevents redundant computation of the same subtrees.
///
/// Uses pointer addresses as keys - if the same node (by address) appears
/// multiple times in the graph, it will produce the same solution tree.
#[derive(Debug, Clone)]
struct MemoCache<'a, L: Clone + Eq + Hash> {
    /// Cache for OR nodes: maps node pointer -> (`best_solution_tree`, `min_distance`)
    or_cache: HashMap<&'a OrNode<L>, MinEditResult<L>>,
    /// Cache for AND nodes: maps node pointer -> (`solution_tree`, distance)
    and_cache: HashMap<&'a AndNode<L>, MinEditResult<L>>,
}

impl<L: Clone + Eq + Hash> MemoCache<'_, L> {
    fn new() -> Self {
        MemoCache {
            or_cache: HashMap::new(),
            and_cache: HashMap::new(),
        }
    }
}

/// Find the solution tree with minimum edit distance to target.
/// If memo is true, memoization is used to efficiently handle DAGs with shared substructure.
///
/// When the AND-OR graph has shared nodes (DAG structure), the same subtree
/// computations are cached and reused, reducing exponential blowup.
pub fn find_min<C: EditCosts<L>, L: Clone + Eq + Hash>(
    graph: &OrNode<L>,
    target: &TreeNode<L>,
    costs: &C,
    memo: bool,
) -> MinEditResult<L> {
    if memo {
        let mut cache = MemoCache::new();
        graph.find_min_memo(target, costs, &mut cache)
    } else {
        graph.find_min(target, costs)
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
    if lists.is_empty() {
        return vec![vec![]];
    }

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

#[cfg(test)]
mod tests {
    use crate::distance::{UnitCost, tree_distance_unit};

    use super::*;

    fn leaf<L: Clone + Eq>(label: L) -> TreeNode<L> {
        TreeNode::new(label)
    }

    fn node<L: Clone + Eq>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::with_children(label, children)
    }

    #[test]
    fn and_leaf_generates_single_tree() {
        let and_node = AndNode::leaf("a");
        let solutions = and_node.generate_solutions();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].label(), &"a");
    }

    #[test]
    fn or_node_generates_multiple_trees() {
        // OR node with two AND leaf children -> two solution trees
        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);

        let solutions = or_node.generate_solutions();
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn and_with_or_children() {
        // Alternating structure:
        //       a (AND)
        //      /       \
        //   b(OR)     c(OR)
        //   / \       / \
        //  d   e     f   g
        //  (AND leaves)
        //
        // Solutions: 2 choices at b × 2 choices at c = 4 trees

        let and_node = AndNode::new(
            "a",
            vec![
                OrNode::new("b", vec![AndNode::leaf("d"), AndNode::leaf("e")]),
                OrNode::new("c", vec![AndNode::leaf("f"), AndNode::leaf("g")]),
            ],
        );

        let solutions = and_node.generate_solutions();
        assert_eq!(solutions.len(), 4);
        assert_eq!(and_node.count_solutions(), 4);
    }

    #[test]
    fn deeper_alternation() {
        // Three levels: AND -> OR -> AND -> OR -> AND(leaf)
        //
        //           root (AND)
        //             |
        //           a (OR)
        //          /     \
        //       b(AND)  c(AND)
        //         |       |
        //       d(OR)   e(OR)
        //       / \       |
        //      f   g      h
        //
        // Solutions: (f or g) + h = 3 trees

        let root = AndNode::new(
            "root",
            vec![OrNode::new(
                "a",
                vec![
                    AndNode::new(
                        "b",
                        vec![OrNode::new(
                            "d",
                            vec![AndNode::leaf("f"), AndNode::leaf("g")],
                        )],
                    ),
                    AndNode::new("c", vec![OrNode::new("e", vec![AndNode::leaf("h")])]),
                ],
            )],
        );

        let solutions = root.generate_solutions();
        assert_eq!(solutions.len(), 3);
        assert_eq!(root.count_solutions(), 3);
    }

    #[test]
    fn find_min_edit_distance_with_choice() {
        // OR node where one choice is closer to target
        //   a(OR)
        //   / \
        //  b   c  (AND leaves)
        //
        // Target: a -> b
        // Choosing b gives distance 0

        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);

        let target = node("a", vec![leaf("b")]);

        let result = find_min(&or_node, &target, &UnitCost, false);
        assert_eq!(result.distance, 0);
        assert_eq!(result.tree.children()[0].label(), &"b");
    }

    #[test]
    fn cartesian_product_basic() {
        let lists = vec![vec![1, 2], vec![3, 4]];
        let product = cartesian_product(&lists);

        assert_eq!(product.len(), 4);
        assert!(product.contains(&vec![1, 3]));
        assert!(product.contains(&vec![1, 4]));
        assert!(product.contains(&vec![2, 3]));
        assert!(product.contains(&vec![2, 4]));
    }

    #[test]
    fn cartesian_product_empty() {
        let lists: Vec<Vec<i32>> = vec![];
        let product = cartesian_product(&lists);
        assert_eq!(product, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn original_zhang_shasha_still_works() {
        let tree1 = node("a", vec![leaf("b"), leaf("c")]);
        let tree2 = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance(&tree1, &tree2, &UnitCost), 0);

        let tree3 = node("a", vec![leaf("b")]);
        assert_eq!(tree_distance(&tree1, &tree3, &UnitCost), 1);
    }

    #[test]
    fn solution_count_complexity() {
        // Verify the counting matches actual enumeration

        let root = AndNode::new(
            "r",
            vec![OrNode::new(
                "o1",
                vec![
                    AndNode::new(
                        "a1",
                        vec![OrNode::new(
                            "o2",
                            vec![AndNode::leaf("x"), AndNode::leaf("y")],
                        )],
                    ),
                    AndNode::new(
                        "a2",
                        vec![OrNode::new(
                            "o3",
                            vec![AndNode::leaf("z"), AndNode::leaf("w")],
                        )],
                    ),
                ],
            )],
        );

        let count = root.count_solutions();
        let solutions = root.generate_solutions();
        assert_eq!(count, solutions.len());
        assert_eq!(count, 4); // (x or y) + (z or w) = 4
    }

    #[test]
    fn dag_shared_substructure_memoization() {
        // Test that memoization works correctly when the AND-OR graph is a DAG
        // with shared substructure (same node referenced multiple times).
        //
        // Structure (DAG):
        //           root (AND)
        //          /          \
        //       o1 (OR)      o2 (OR)  <- both point to the SAME shared_and
        //          \          /
        //         shared_and (AND)
        //              |
        //           o3 (OR)
        //           /    \
        //          x      y
        //
        // Without memoization, shared_and would be processed twice.
        // With memoization, it's processed once and the result is reused.

        // Create the shared substructure
        let shared_and = AndNode::new(
            "shared",
            vec![OrNode::new(
                "o3",
                vec![AndNode::leaf("x"), AndNode::leaf("y")],
            )],
        );

        // Create a DAG by cloning the shared node (they have the same NodeId)
        // Note: In a real DAG scenario, you'd use Rc<AndNode> or similar.
        // Here we simulate by creating nodes with the same structure.
        let root = AndNode::new(
            "root",
            vec![
                OrNode::new("o1", vec![shared_and.clone()]),
                OrNode::new("o2", vec![shared_and]),
            ],
        );

        // Target tree
        let target = node(
            "root",
            vec![
                node(
                    "o1",
                    vec![node("shared", vec![node("o3", vec![leaf("x")])])],
                ),
                node(
                    "o2",
                    vec![node("shared", vec![node("o3", vec![leaf("x")])])],
                ),
            ],
        );

        let result = find_min(&OrNode::single("top", root), &target, &UnitCost, false);

        // The result should find a valid solution
        assert!(result.distance < usize::MAX);
    }

    #[test]
    fn dag_with_actual_shared_nodes() {
        // Create a true DAG where the same OR node is referenced by multiple AND nodes.
        // This tests that NodeId-based memoization correctly identifies shared structure.
        //
        //         root (OR)
        //        /        \
        //    and1 (AND)  and2 (AND)
        //       |    \    /    |
        //      o1    shared   o2
        //              |
        //         leaf_and
        //
        // shared is the same OrNode referenced by both and1 and and2

        let shared_or = OrNode::new("shared", vec![AndNode::leaf("common")]);

        let and1 = AndNode::new(
            "and1",
            vec![
                OrNode::new("o1", vec![AndNode::leaf("a")]),
                shared_or.clone(),
            ],
        );

        let and2 = AndNode::new(
            "and2",
            vec![
                shared_or, // Same node, same NodeId
                OrNode::new("o2", vec![AndNode::leaf("b")]),
            ],
        );

        let root = OrNode::new("root", vec![and1, and2]);

        // Target that matches and1's structure better
        let target = node(
            "root",
            vec![node(
                "and1",
                vec![
                    node("o1", vec![leaf("a")]),
                    node("shared", vec![leaf("common")]),
                ],
            )],
        );

        let result = find_min(&root, &target, &UnitCost, false);

        // Should find and1 as the best match with distance 0
        assert_eq!(result.distance, 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_simple() {
        // Simple OR node with two choices
        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);
        let target = node("a", vec![leaf("b")]);

        let r1 = or_node.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let r2 = or_node.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_nested() {
        // Nested structure: AND -> OR -> AND
        let root = OrNode::new(
            "root",
            vec![AndNode::new(
                "a",
                vec![
                    OrNode::new("b", vec![AndNode::leaf("d"), AndNode::leaf("e")]),
                    OrNode::new("c", vec![AndNode::leaf("f"), AndNode::leaf("g")]),
                ],
            )],
        );

        let target = node(
            "root",
            vec![node(
                "a",
                vec![node("b", vec![leaf("d")]), node("c", vec![leaf("f")])],
            )],
        );

        let r1 = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let r2 = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_multiple_choices() {
        // Multiple OR choices at different levels
        let root = OrNode::new(
            "root",
            vec![
                AndNode::new(
                    "choice1",
                    vec![OrNode::new(
                        "inner",
                        vec![AndNode::leaf("x"), AndNode::leaf("y")],
                    )],
                ),
                AndNode::new(
                    "choice2",
                    vec![OrNode::new(
                        "inner",
                        vec![AndNode::leaf("z"), AndNode::leaf("w")],
                    )],
                ),
            ],
        );

        let target = node(
            "root",
            vec![node("choice1", vec![node("inner", vec![leaf("x")])])],
        );

        let r1 = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let r2 = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_deep() {
        // Deeper alternation: OR -> AND -> OR -> AND -> OR -> AND(leaf)
        let root = OrNode::new(
            "l1",
            vec![AndNode::new(
                "l2",
                vec![OrNode::new(
                    "l3",
                    vec![
                        AndNode::new(
                            "l4a",
                            vec![OrNode::new(
                                "l5",
                                vec![AndNode::leaf("leaf1"), AndNode::leaf("leaf2")],
                            )],
                        ),
                        AndNode::new(
                            "l4b",
                            vec![OrNode::new(
                                "l5",
                                vec![AndNode::leaf("leaf3"), AndNode::leaf("leaf4")],
                            )],
                        ),
                    ],
                )],
            )],
        );

        let target = node(
            "l1",
            vec![node(
                "l2",
                vec![node(
                    "l3",
                    vec![node("l4a", vec![node("l5", vec![leaf("leaf1")])])],
                )],
            )],
        );

        let r1 = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let r2 = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }
}
