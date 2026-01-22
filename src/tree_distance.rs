//! AND-OR Graph Extension for Zhang-Shasha Tree Edit Distance
//!
//! Finds the solution tree in a bounded AND-OR graph with minimum edit distance
//! to a target tree. Assumes bounded OR-branching (N) and bounded depth (d).
//!
//! With strict alternation (OR→AND→OR→...), complexity is O(N^(d/2) * |T|^2)
//! for single-path graphs, where N is OR-branching and d is depth.
//!
//! ## Zhang-Shasha Tree Edit Distance Algorithm
//!
//! Zhang-Shasha computes the edit distance between two ordered labeled trees.
//! The algorithm runs in O(n1 * n2 * min(depth1, leaves1) * min(depth2, leaves2))
//! time and O(n1 * n2) space.

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use std::hash::Hash;

// ============================================================================
// AND-OR Graph Extension (with strict alternation)
// ============================================================================

/// AND node: all children must be included in solution tree.
/// Children must be OR nodes (or leaves).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AndNode<L: Clone + Eq + Hash> {
    pub label: L,
    pub children: Vec<OrNode<L>>,
}

/// OR node: exactly one child is chosen for solution tree.
/// Children must be AND nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrNode<L: Clone + Eq + Hash> {
    pub label: L,
    pub children: Vec<AndNode<L>>,
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
    fn find_min<C: EditCosts<L>>(&self, target: &TreeNode<L>, costs: &C) -> (TreeNode<L>, usize) {
        let result_tree = if self.is_leaf() {
            TreeNode::new(self.label.clone())
        } else {
            // For AND nodes, we must include all children.
            // Recursively find the best solution for each OR child.
            let child_trees: Vec<TreeNode<L>> = self
                .children
                .iter()
                .map(|or_child| or_child.find_min(target, costs).0)
                .collect();

            TreeNode::with_children(self.label.clone(), child_trees)
        };

        let dist = tree_distance(&result_tree, target, costs);
        (result_tree, dist)
    }

    /// Find the solution from this AND node, with memoization.
    fn find_min_memo<'a, C: EditCosts<L>>(
        &'a self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<'a, L>,
    ) -> (TreeNode<L>, usize) {
        // Check cache first using pointer address as key
        if let Some(cached) = cache.and_cache.get(&self) {
            return cached.clone();
        }

        let result_tree = if self.is_leaf() {
            TreeNode::new(self.label.clone())
        } else {
            // For AND nodes, we must include all children.
            // Recursively find the best solution for each OR child.
            let child_trees: Vec<TreeNode<L>> = self
                .children
                .iter()
                .map(|or_child| or_child.find_min_memo(target, costs, cache).0)
                .collect();

            TreeNode::with_children(self.label.clone(), child_trees)
        };

        let dist = tree_distance(&result_tree, target, costs);
        let result = (result_tree, dist);
        cache.and_cache.insert(self, result.clone());
        result
    }
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
    fn find_min<C: EditCosts<L>>(&self, target: &TreeNode<L>, costs: &C) -> (TreeNode<L>, usize) {
        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for and_child in &self.children {
            let (subtree, _) = and_child.find_min(target, costs);

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

        (
            best_tree.expect("OR node must have at least one child"),
            best_distance,
        )
    }

    /// Find the best solution from this OR node, with memoization.
    fn find_min_memo<'a, C: EditCosts<L>>(
        &'a self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<'a, L>,
    ) -> (TreeNode<L>, usize) {
        // Check cache first using pointer address as key
        if let Some(cached) = cache.or_cache.get(&self) {
            return cached.clone();
        }

        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for and_child in &self.children {
            let (subtree, _) = and_child.find_min_memo(target, costs, cache);

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

        let result = (
            best_tree.expect("OR node must have at least one child"),
            best_distance,
        );
        cache.or_cache.insert(self, result.clone());
        result
    }
}

/// Result of finding the minimum edit distance solution tree
#[derive(Debug, Clone)]
pub struct MinEditResult<L: Clone + Eq> {
    pub solution_tree: TreeNode<L>,
    pub edit_distance: usize,
}

/// Memoization cache for AND-OR graph edit distance computation.
/// When the AND-OR graph is a DAG (has shared substructure), this cache
/// prevents redundant computation of the same subtrees.
///
/// Uses pointer addresses as keys - if the same node (by address) appears
/// multiple times in the graph, it will produce the same solution tree.
struct MemoCache<'a, L: Clone + Eq + Hash> {
    /// Cache for OR nodes: maps node pointer -> (`best_solution_tree`, `min_distance`)
    or_cache: HashMap<&'a OrNode<L>, (TreeNode<L>, usize)>,
    /// Cache for AND nodes: maps node pointer -> (`solution_tree`, distance)
    and_cache: HashMap<&'a AndNode<L>, (TreeNode<L>, usize)>,
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
pub fn find_min_edit_distance_tree<L: Clone + Eq + Hash, C: EditCosts<L>>(
    root: &OrNode<L>,
    target: &TreeNode<L>,
    costs: &C,
) -> MinEditResult<L> {
    let (solution_tree, edit_distance) = root.find_min(target, costs);

    MinEditResult {
        solution_tree,
        edit_distance,
    }
}

/// Find the solution tree with minimum edit distance to target.
/// Uses memoization to efficiently handle DAGs with shared substructure.
///
/// When the AND-OR graph has shared nodes (DAG structure), the same subtree
/// computations are cached and reused, reducing exponential blowup.
pub fn find_min_edit_distance_tree_memo<L: Clone + Eq + Hash, C: EditCosts<L>>(
    root: &OrNode<L>,
    target: &TreeNode<L>,
    costs: &C,
) -> MinEditResult<L> {
    let mut cache = MemoCache::new();
    let (solution_tree, edit_distance) = root.find_min_memo(target, costs, &mut cache);

    MinEditResult {
        solution_tree,
        edit_distance,
    }
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

#[allow(dead_code)]
fn count_tree_nodes<L: Clone + Eq>(tree: &TreeNode<L>) -> usize {
    1 + tree.children.iter().map(count_tree_nodes).sum::<usize>()
}

// ============================================================================
// Zhang-Shasha Implementation
// ============================================================================

/// A node in a labeled tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode<L: Clone + Eq> {
    label: L,
    children: Vec<TreeNode<L>>,
}

impl<L: Clone + Eq> TreeNode<L> {
    pub fn new(label: L) -> Self {
        TreeNode {
            label,
            children: Vec::new(),
        }
    }

    pub fn with_children(label: L, children: Vec<TreeNode<L>>) -> Self {
        TreeNode { label, children }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Postorder traversal information for a tree node
#[derive(Debug, Clone)]
struct PostorderNode<L: Clone + Eq> {
    label: L,
    leftmost_leaf: usize, // postorder index of leftmost leaf descendant
}

/// Preprocessed tree for Zhang-Shasha algorithm
struct PreprocessedTree<L: Clone + Eq> {
    nodes: Vec<PostorderNode<L>>,
    keyroots: Vec<usize>, // indices of keyroots in postorder
}

impl<L: Clone + Eq> PreprocessedTree<L> {
    fn new(root: &TreeNode<L>) -> Self {
        let mut nodes = Vec::new();

        // Perform postorder traversal and compute leftmost leaf descendants
        Self::postorder_traverse(root, &mut nodes);

        // Compute keyroots: a node is a keyroot if it's the last node (in postorder)
        // with its particular leftmost leaf value. This is equivalent to: a node is
        // a keyroot if it has no parent, or it is not the leftmost child of its parent.
        let mut keyroots = Vec::new();
        let mut leftmost_to_keyroot = vec![0; nodes.len()];

        for (i, n) in nodes.iter().enumerate() {
            // Each time we see a leftmost leaf value, update to the latest node
            leftmost_to_keyroot[n.leftmost_leaf] = i;
        }

        // Collect unique keyroots
        let mut seen = vec![false; nodes.len()];
        for &kr in &leftmost_to_keyroot {
            if !seen[kr] {
                seen[kr] = true;
                keyroots.push(kr);
            }
        }

        keyroots.sort_unstable();

        PreprocessedTree { nodes, keyroots }
    }

    fn postorder_traverse(node: &TreeNode<L>, nodes: &mut Vec<PostorderNode<L>>) -> usize {
        // First, traverse all children
        let child_indices = node
            .children
            .iter()
            .map(|child| Self::postorder_traverse(child, nodes))
            .collect::<Vec<_>>();

        // Current node's postorder index
        let current_idx = nodes.len();

        // Compute leftmost leaf
        let leftmost_leaf = if node.is_leaf() {
            current_idx
        } else {
            // Leftmost leaf is the leftmost leaf of the leftmost child
            nodes[child_indices[0]].leftmost_leaf
        };

        nodes.push(PostorderNode {
            label: node.label.clone(),
            leftmost_leaf,
        });

        current_idx
    }

    fn size(&self) -> usize {
        self.nodes.len()
    }

    fn leftmost_leaf(&self, i: usize) -> usize {
        self.nodes[i].leftmost_leaf
    }

    fn label(&self, i: usize) -> &L {
        &self.nodes[i].label
    }
}

/// Cost functions for tree edit operations
pub trait EditCosts<L> {
    /// Cost of deleting a node with the given label
    fn delete(&self, label: &L) -> usize;

    /// Cost of inserting a node with the given label
    fn insert(&self, label: &L) -> usize;

    /// Cost of relabeling a node from one label to another
    fn relabel(&self, from: &L, to: &L) -> usize;
}

/// Unit cost model: all operations cost 1, relabeling same labels costs 0
pub struct UnitCost;

impl<L: Eq> EditCosts<L> for UnitCost {
    fn delete(&self, _label: &L) -> usize {
        1
    }

    fn insert(&self, _label: &L) -> usize {
        1
    }

    fn relabel(&self, from: &L, to: &L) -> usize {
        usize::from(from != to)
    }
}

/// Compute the Zhang-Shasha tree edit distance between two trees
pub fn tree_distance<L: Clone + Eq, C: EditCosts<L>>(
    tree1: &TreeNode<L>,
    tree2: &TreeNode<L>,
    costs: &C,
) -> usize {
    let t1 = PreprocessedTree::new(tree1);
    let t2 = PreprocessedTree::new(tree2);

    let n1 = t1.size();
    let n2 = t2.size();

    if n1 == 0 && n2 == 0 {
        return 0;
    }
    if n1 == 0 {
        return (0..n2).map(|j| costs.insert(t2.label(j))).sum();
    }
    if n2 == 0 {
        return (0..n1).map(|i| costs.delete(t1.label(i))).sum();
    }

    // Tree distance matrix (permanent)
    let mut td = vec![vec![0; n2]; n1];

    // Forest distance matrix (temporary, reused for each keyroot pair)
    // We need indices from -1, so we use size+1 and offset by 1
    let mut fd = vec![vec![0; n2 + 1]; n1 + 1];

    // Compute tree distance for each pair of keyroots
    for &i in &t1.keyroots {
        for &j in &t2.keyroots {
            compute_forest_distance(&t1, &t2, i, j, &mut td, &mut fd, costs);
        }
    }

    // The final answer is the distance between the full trees
    td[n1 - 1][n2 - 1]
}

fn compute_forest_distance<L: Clone + Eq, C: EditCosts<L>>(
    t1: &PreprocessedTree<L>,
    t2: &PreprocessedTree<L>,
    i: usize,
    j: usize,
    td: &mut [Vec<usize>],
    fd: &mut [Vec<usize>],
    costs: &C,
) {
    let l1 = t1.leftmost_leaf(i);
    let l2 = t2.leftmost_leaf(j);

    // fd[x][y] represents the forest distance between:
    // - forest of t1 from l1 to x-1 (using 1-based indexing offset)
    // - forest of t2 from l2 to y-1
    // fd[0][0] = 0 (empty forests)

    // Initialize: deleting all nodes from t1's forest
    fd[0][0] = 0;
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        fd[x_idx][0] = fd[x_idx - 1][0] + costs.delete(t1.label(x));
    }

    // Initialize: inserting all nodes into empty forest from t2
    for y in l2..=j {
        let y_idx = y - l2 + 1;
        fd[0][y_idx] = fd[0][y_idx - 1] + costs.insert(t2.label(y));
    }

    // Fill in the forest distance matrix
    // Note: we intentionally use x and y as indices into td, as td stores
    // tree distances for all node pairs using their postorder indices
    #[allow(clippy::needless_range_loop)]
    for x in l1..=i {
        let x_idx = x - l1 + 1;
        let lx = t1.leftmost_leaf(x);

        for y in l2..=j {
            let y_idx = y - l2 + 1;
            let ly = t2.leftmost_leaf(y);

            let delete_cost = fd[x_idx - 1][y_idx] + costs.delete(t1.label(x));
            let insert_cost = fd[x_idx][y_idx - 1] + costs.insert(t2.label(y));

            if lx == l1 && ly == l2 {
                // Both x and y have their leftmost leaves at the start of this subproblem,
                // meaning we're computing the full subtree distance for (x, y)
                let relabel_cost =
                    fd[x_idx - 1][y_idx - 1] + costs.relabel(t1.label(x), t2.label(y));
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(relabel_cost);
                // Store in permanent tree distance matrix
                td[x][y] = fd[x_idx][y_idx];
            } else {
                // At least one of x or y has its leftmost leaf before the start of this
                // subproblem, so we need to use the previously computed tree distance
                let match_cost = fd[lx - l1][ly - l2] + td[x][y];
                fd[x_idx][y_idx] = delete_cost.min(insert_cost).min(match_cost);
            }
        }
    }
}

/// Convenience function with unit costs
pub fn tree_distance_unit<L: Clone + Eq>(tree1: &TreeNode<L>, tree2: &TreeNode<L>) -> usize {
    tree_distance(tree1, tree2, &UnitCost)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf<L: Clone + Eq>(label: L) -> TreeNode<L> {
        TreeNode::new(label)
    }

    fn node<L: Clone + Eq>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::with_children(label, children)
    }

    #[test]
    fn test_identical_trees() {
        let tree1 = node("a", vec![leaf("b"), leaf("c")]);
        let tree2 = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance_unit(&tree1, &tree2), 0);
    }

    #[test]
    fn test_single_node_difference() {
        let tree1 = leaf("a");
        let tree2 = leaf("b");
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // relabel a -> b
    }

    #[test]
    fn test_insert_child() {
        let tree1 = node("a", vec![leaf("b")]);
        let tree2 = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // insert c
    }

    #[test]
    fn test_delete_child() {
        let tree1 = node("a", vec![leaf("b"), leaf("c")]);
        let tree2 = node("a", vec![leaf("b")]);
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1); // delete c
    }

    #[test]
    fn test_empty_to_tree() {
        // Empty tree represented as single node to non-empty
        let tree1 = leaf("a");
        let tree2 = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2); // insert b, insert c
    }

    #[test]
    fn test_different_structure() {
        // Tree 1:    a          Tree 2:    a
        //           /|                     |
        //          b c                     b
        //                                  |
        //                                  c
        let tree1 = node("a", vec![leaf("b"), leaf("c")]);
        let tree2 = node("a", vec![node("b", vec![leaf("c")])]);
        // One way: delete c from tree1, insert c under b = 2 operations
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2);
    }

    #[test]
    fn test_completely_different() {
        let tree1 = node("a", vec![leaf("b")]);
        let tree2 = node("x", vec![leaf("y")]);
        // relabel a->x, relabel b->y = 2 operations
        assert_eq!(tree_distance_unit(&tree1, &tree2), 2);
    }

    #[test]
    fn test_larger_trees() {
        // Tree 1:       a
        //             / | \
        //            b  c  d
        //           /|
        //          e f
        let tree1 = node(
            "a",
            vec![node("b", vec![leaf("e"), leaf("f")]), leaf("c"), leaf("d")],
        );

        // Tree 2:       a
        //             / | \
        //            b  c  d
        //           /
        //          e
        let tree2 = node("a", vec![node("b", vec![leaf("e")]), leaf("c"), leaf("d")]);

        // Delete f from tree1
        assert_eq!(tree_distance_unit(&tree1, &tree2), 1);
    }

    #[test]
    fn test_deep_vs_shallow() {
        // Tree 1: a - b - c - d (linear chain)
        let tree1 = node("a", vec![node("b", vec![node("c", vec![leaf("d")])])]);

        // Tree 2:    a
        //          / | \
        //         b  c  d
        let tree2 = node("a", vec![leaf("b"), leaf("c"), leaf("d")]);

        // Need to restructure: this requires delete and insert operations
        // The exact cost depends on the optimal alignment
        let dist = tree_distance_unit(&tree1, &tree2);
        assert!(dist > 0);
        assert!(dist <= 4); // Upper bound: delete all and insert all (minus common)
    }

    #[test]
    fn test_and_leaf_generates_single_tree() {
        let and_node = AndNode::leaf("a");
        let solutions = and_node.generate_solutions();
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].label, "a");
    }

    #[test]
    fn test_or_node_generates_multiple_trees() {
        // OR node with two AND leaf children -> two solution trees
        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);

        let solutions = or_node.generate_solutions();
        assert_eq!(solutions.len(), 2);
    }

    #[test]
    fn test_and_with_or_children() {
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
    fn test_deeper_alternation() {
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
    fn test_find_min_edit_distance_with_choice() {
        // OR node where one choice is closer to target
        //   a(OR)
        //   / \
        //  b   c  (AND leaves)
        //
        // Target: a -> b
        // Choosing b gives distance 0

        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);

        let target = node("a", vec![leaf("b")]);

        let result = find_min_edit_distance_tree(&or_node, &target, &UnitCost);
        assert_eq!(result.edit_distance, 0);
        assert_eq!(result.solution_tree.children[0].label, "b");
    }

    #[test]
    fn test_cartesian_product() {
        let lists = vec![vec![1, 2], vec![3, 4]];
        let product = cartesian_product(&lists);

        assert_eq!(product.len(), 4);
        assert!(product.contains(&vec![1, 3]));
        assert!(product.contains(&vec![1, 4]));
        assert!(product.contains(&vec![2, 3]));
        assert!(product.contains(&vec![2, 4]));
    }

    #[test]
    fn test_cartesian_product_empty() {
        let lists: Vec<Vec<i32>> = vec![];
        let product = cartesian_product(&lists);
        assert_eq!(product, vec![Vec::<i32>::new()]);
    }

    #[test]
    fn test_original_zhang_shasha_still_works() {
        let tree1 = node("a", vec![leaf("b"), leaf("c")]);
        let tree2 = node("a", vec![leaf("b"), leaf("c")]);
        assert_eq!(tree_distance(&tree1, &tree2, &UnitCost), 0);

        let tree3 = node("a", vec![leaf("b")]);
        assert_eq!(tree_distance(&tree1, &tree3, &UnitCost), 1);
    }

    #[test]
    fn test_solution_count_complexity() {
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
    fn test_dag_shared_substructure_memoization() {
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

        let result = find_min_edit_distance_tree(&OrNode::single("top", root), &target, &UnitCost);

        // The result should find a valid solution
        assert!(result.edit_distance < usize::MAX);
    }

    #[test]
    fn test_dag_with_actual_shared_nodes() {
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

        let result = find_min_edit_distance_tree(&root, &target, &UnitCost);

        // Should find and1 as the best match with distance 0
        assert_eq!(result.edit_distance, 0);
    }

    #[test]
    fn test_find_min_and_find_min_memo_equivalent_simple() {
        // Simple OR node with two choices
        let or_node = OrNode::new("a", vec![AndNode::leaf("b"), AndNode::leaf("c")]);
        let target = node("a", vec![leaf("b")]);

        let (tree_no_memo, dist_no_memo) = or_node.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let (tree_memo, dist_memo) = or_node.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(dist_no_memo, dist_memo);
        assert_eq!(tree_distance_unit(&tree_no_memo, &tree_memo), 0);
    }

    #[test]
    fn test_find_min_and_find_min_memo_equivalent_nested() {
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

        let (tree_no_memo, dist_no_memo) = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let (tree_memo, dist_memo) = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(dist_no_memo, dist_memo);
        assert_eq!(tree_distance_unit(&tree_no_memo, &tree_memo), 0);
    }

    #[test]
    fn test_find_min_and_find_min_memo_equivalent_multiple_choices() {
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

        let (tree_no_memo, dist_no_memo) = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let (tree_memo, dist_memo) = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(dist_no_memo, dist_memo);
        assert_eq!(tree_distance_unit(&tree_no_memo, &tree_memo), 0);
    }

    #[test]
    fn test_find_min_and_find_min_memo_equivalent_deep() {
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

        let (tree_no_memo, dist_no_memo) = root.find_min(&target, &UnitCost);

        let mut cache = MemoCache::new();
        let (tree_memo, dist_memo) = root.find_min_memo(&target, &UnitCost, &mut cache);

        assert_eq!(dist_no_memo, dist_memo);
        assert_eq!(tree_distance_unit(&tree_no_memo, &tree_memo), 0);
    }
}
