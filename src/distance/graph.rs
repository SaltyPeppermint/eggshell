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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AndOrGraph<L: Clone + Eq + Hash> {
    or: Vec<OrNode<L>>,
    and: Vec<AndNode<L>>,
    root: OrId,
}

impl<L: Clone + Eq + Hash> AndOrGraph<L> {
    #[must_use]
    pub fn new(or: Vec<OrNode<L>>, and: Vec<AndNode<L>>, root: OrId) -> Self {
        Self { or, and, root }
    }

    #[must_use]
    pub fn or(&self, id: OrId) -> &OrNode<L> {
        &self.or[id.0]
    }

    #[must_use]
    pub fn and(&self, id: AndId) -> &AndNode<L> {
        &self.and[id.0]
    }

    /// Find the solution tree with minimum edit distance to target.
    /// No cycles are allowed
    #[must_use]
    pub fn find_min<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
    ) -> Option<MinEditResult<L>> {
        self.find_min_cyclic(target, costs, 0)
    }

    /// Find the solution tree with minimum edit distance to target, with memoization.
    /// No cycles are allowed
    ///
    /// When the AND-OR graph has shared nodes (DAG structure), the same subtree
    /// computations are cached and reused, reducing exponential blowup.
    #[must_use]
    pub fn find_min_memo<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
    ) -> Option<MinEditResult<L>> {
        self.find_min_memo_cyclic(target, costs, 0)
    }

    /// Find the solution tree with minimum edit distance to target, with cycle handling.
    ///
    /// This variant handles graphs with cycles by tracking the current path and
    /// limiting how many times each node may be revisited.
    /// Returns `Some(result)` if a valid solution was found, `None` if all paths hit cycle limits
    #[must_use]
    pub fn find_min_cyclic<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        max_revisits: usize,
    ) -> Option<MinEditResult<L>> {
        let mut path = PathTracker::new(max_revisits);
        if !path.can_visit(self.root) {
            return None;
        }
        path.enter(self.root);
        let result = self.or(self.root).find_min(target, costs, &mut path, self);
        path.leave(self.root);
        result
    }

    /// Find the solution tree with minimum edit distance to target, with memoization
    /// and multiple cycle visits.
    #[must_use]
    pub fn find_min_memo_cyclic<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        max_revisits: usize,
    ) -> Option<MinEditResult<L>> {
        let mut cache = MemoCache::new();
        let mut path = PathTracker::new(max_revisits);
        if !path.can_visit(self.root) {
            return None;
        }
        path.enter(self.root);
        let result = self
            .or(self.root)
            .find_min_memo(target, costs, &mut cache, &mut path, self);
        path.leave(self.root);
        result
    }
}

/// AND node: all children must be included in solution tree.
/// Children must be OR nodes (or leaves).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AndNode<L: Clone + Eq + Hash> {
    label: L,
    children: Vec<OrId>,
}

impl<L: Clone + Eq + Hash> AndNode<L> {
    /// AND node with OR children
    pub fn new(label: L, children: Vec<OrId>) -> Self {
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

    pub fn children(&self) -> &[OrId] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Generate all valid solution trees from this AND node.
    /// Includes all OR children, taking cartesian product of their choices.
    pub fn generate_solutions(&self, graph: &AndOrGraph<L>) -> Vec<TreeNode<L>> {
        if self.is_leaf() {
            return vec![TreeNode::new(self.label.clone())];
        }

        // Each OR child contributes a set of possible subtrees
        let child_solutions: Vec<Vec<TreeNode<L>>> = self
            .children
            .iter()
            .map(|i| graph.or(*i))
            .map(|or_node| or_node.generate_solutions(graph))
            .collect();

        cartesian_product(&child_solutions)
            .into_iter()
            .map(|children| TreeNode::with_children(self.label.clone(), children))
            .collect()
    }

    /// Count the number of solution trees (for complexity analysis)
    pub fn count_solutions(&self, graph: &AndOrGraph<L>) -> usize {
        if self.is_leaf() {
            return 1;
        }
        self.children
            .iter()
            .map(|i| graph.or(*i))
            .map(|or_node| or_node.count_solutions(graph))
            .product()
    }

    // /// Find the solution from this AND node without memoization.
    // fn find_min<C: EditCosts<L>>(
    //     &self,
    //     target: &TreeNode<L>,
    //     costs: &C,
    //     graph: &AndOrGraph<L>,
    // ) -> MinEditResult<L> {
    //     let tree = if self.is_leaf() {
    //         TreeNode::new(self.label.clone())
    //     } else {
    //         // For AND nodes, we must include all children.
    //         // Recursively find the best solution for each OR child.
    //         let child_trees: Vec<TreeNode<L>> = self
    //             .children
    //             .iter()
    //             .map(|i| graph.or(*i))
    //             .map(|or_child| or_child.find_min(target, costs, graph).tree)
    //             .collect();

    //         TreeNode::with_children(self.label.clone(), child_trees)
    //     };

    //     let distance = tree_distance(&tree, target, costs);
    //     MinEditResult { tree, distance }
    // }

    // /// Find the solution from this AND node, with memoization.
    // fn find_min_memo<C: EditCosts<L>>(
    //     &self,
    //     target: &TreeNode<L>,
    //     costs: &C,
    //     cache: &mut MemoCache<L>,
    //     graph: &AndOrGraph<L>,
    // ) -> MinEditResult<L> {
    //     let tree = if self.is_leaf() {
    //         TreeNode::new(self.label.clone())
    //     } else {
    //         // For AND nodes, we must include all children.
    //         // Recursively find the best solution for each OR child.
    //         let child_trees: Vec<TreeNode<L>> = self
    //             .children
    //             .iter()
    //             .map(|&or_id| {
    //                 // Check cache first
    //                 if let Some(c) = cache.or_cache.get(&or_id) {
    //                     c.tree.clone()
    //                 } else {
    //                     let result = graph.or(or_id).find_min_memo(target, costs, cache, graph);
    //                     cache.or_cache.insert(or_id, result.clone());
    //                     result.tree
    //                 }
    //             })
    //             .collect();

    //         TreeNode::with_children(self.label.clone(), child_trees)
    //     };

    //     let distance = tree_distance(&tree, target, costs);
    //     MinEditResult { tree, distance }
    // }

    /// Find the solution from this AND node with cycle detection.
    /// Caller must have already checked and entered this node in the path tracker.
    fn find_min<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        path: &mut PathTracker,
        graph: &AndOrGraph<L>,
    ) -> Option<MinEditResult<L>> {
        if self.is_leaf() {
            let tree = TreeNode::new(self.label.clone());
            let distance = tree_distance(&tree, target, costs);
            return Some(MinEditResult { tree, distance });
        }

        // For AND nodes, we must include all children.
        // If any child fails due to cycle limits, this AND node fails.
        let mut child_trees = Vec::with_capacity(self.children.len());

        for &or_id in &self.children {
            if !path.can_visit(or_id) {
                return None;
            }
            path.enter(or_id);
            let result = graph.or(or_id).find_min(target, costs, path, graph);
            path.leave(or_id);

            if let Some(r) = result {
                child_trees.push(r.tree);
            } else {
                return None;
            }
        }

        let tree = TreeNode::with_children(self.label.clone(), child_trees);
        let distance = tree_distance(&tree, target, costs);
        Some(MinEditResult { tree, distance })
    }

    /// Find the solution from this AND node with memoization and cycle detection.
    /// Caller must have already checked and entered this node in the path tracker.
    fn find_min_memo<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<L>,
        path: &mut PathTracker,
        graph: &AndOrGraph<L>,
    ) -> Option<MinEditResult<L>> {
        if self.is_leaf() {
            let tree = TreeNode::new(self.label.clone());
            let distance = tree_distance(&tree, target, costs);
            return Some(MinEditResult { tree, distance });
        }

        // For AND nodes, we must include all children.
        let mut child_trees = Vec::with_capacity(self.children.len());

        for &or_id in &self.children {
            // Check cache first, but still need to validate path
            let child_result = if let Some(cached) = cache.or_cache.get(&or_id) {
                // Even with cache hit, check if we can visit this node on current path
                if path.can_visit(or_id) {
                    Some(cached.clone())
                } else {
                    None
                }
            } else if path.can_visit(or_id) {
                path.enter(or_id);
                let result = graph
                    .or(or_id)
                    .find_min_memo(target, costs, cache, path, graph);
                path.leave(or_id);
                if let Some(ref r) = result {
                    cache.or_cache.insert(or_id, r.clone());
                }
                result
            } else {
                None
            };

            if let Some(r) = child_result {
                child_trees.push(r.tree);
            } else {
                return None;
            }
        }

        let tree = TreeNode::with_children(self.label.clone(), child_trees);
        let distance = tree_distance(&tree, target, costs);
        Some(MinEditResult { tree, distance })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AndId(usize);

impl AndId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl From<AndId> for usize {
    fn from(value: AndId) -> Self {
        value.0
    }
}

/// OR node: exactly one child is chosen for solution tree.
/// Children must be AND nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrNode<L: Clone + Eq + Hash> {
    label: L,
    children: Vec<AndId>,
}

impl<L: Clone + Eq + Hash> OrNode<L> {
    /// OR node with AND children
    #[expect(clippy::missing_panics_doc)]
    pub fn new(label: L, children: Vec<AndId>) -> Self {
        assert!(!children.is_empty(), "OR node must have at least one child");
        OrNode { label, children }
    }

    /// Single-choice OR node (wraps an AND node)
    pub fn single(label: L, child: AndId) -> Self {
        OrNode {
            label,
            children: vec![child],
        }
    }

    pub fn children(&self) -> &[AndId] {
        &self.children
    }

    pub fn label(&self) -> &L {
        &self.label
    }

    /// Generate all valid solution trees from this OR node.
    /// Chooses exactly one AND child.
    pub fn generate_solutions(&self, graph: &AndOrGraph<L>) -> Vec<TreeNode<L>> {
        // Each AND child is a possible choice; collect all their solutions
        self.children
            .iter()
            .map(|i| graph.and(*i))
            .flat_map(|and_child| {
                and_child
                    .generate_solutions(graph)
                    .into_iter()
                    .map(|subtree| {
                        // OR node label becomes parent of the chosen subtree
                        TreeNode::with_children(self.label.clone(), vec![subtree])
                    })
            })
            .collect()
    }

    /// Count the number of solution trees
    pub fn count_solutions(&self, graph: &AndOrGraph<L>) -> usize {
        self.children
            .iter()
            .map(|i| graph.and(*i))
            .map(|and_child| and_child.count_solutions(graph))
            .sum()
    }

    /// Find the best solution from this OR node with cycle detection.
    /// Caller must have already checked and entered this node in the path tracker.
    fn find_min<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        path: &mut PathTracker,
        graph: &AndOrGraph<L>,
    ) -> Option<MinEditResult<L>> {
        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for &and_id in &self.children {
            let result = graph.and(and_id).find_min(target, costs, path, graph);

            if let Some(r) = result {
                // OR node label becomes parent of the chosen subtree
                let candidate_tree = TreeNode::with_children(self.label.clone(), vec![r.tree]);
                let dist = tree_distance(&candidate_tree, target, costs);

                if dist < best_distance {
                    best_distance = dist;
                    best_tree = Some(candidate_tree);
                }
                if best_distance == 0 {
                    break;
                }
            }
        }

        best_tree.map(|tree| MinEditResult {
            tree,
            distance: best_distance,
        })
    }

    /// Find the best solution from this OR node with memoization and cycle detection.
    /// Caller must have already checked and entered this node in the path tracker.
    fn find_min_memo<C: EditCosts<L>>(
        &self,
        target: &TreeNode<L>,
        costs: &C,
        cache: &mut MemoCache<L>,
        path: &mut PathTracker,
        graph: &AndOrGraph<L>,
    ) -> Option<MinEditResult<L>> {
        let mut best_distance = usize::MAX;
        let mut best_tree = None;

        // Try each AND child and find the one with minimum edit distance
        for &and_id in &self.children {
            // Check cache first
            let child_result = if let Some(cached) = cache.and_cache.get(&and_id) {
                Some(cached.clone())
            } else {
                let result = graph
                    .and(and_id)
                    .find_min_memo(target, costs, cache, path, graph);
                if let Some(ref r) = result {
                    cache.and_cache.insert(and_id, r.clone());
                }
                result
            };

            if let Some(r) = child_result {
                // OR node label becomes parent of the chosen subtree
                let candidate_tree = TreeNode::with_children(self.label.clone(), vec![r.tree]);
                let dist = tree_distance(&candidate_tree, target, costs);

                if dist < best_distance {
                    best_distance = dist;
                    best_tree = Some(candidate_tree);
                }
                if best_distance == 0 {
                    break;
                }
            }
        }

        best_tree.map(|tree| MinEditResult {
            tree,
            distance: best_distance,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrId(usize);

impl OrId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl From<OrId> for usize {
    fn from(value: OrId) -> Self {
        value.0
    }
}

/// Memoization cache for AND-OR graph edit distance computation.
/// When the AND-OR graph is a DAG (has shared substructure), this cache
/// prevents redundant computation of the same subtrees.
///
/// Uses node IDs as keys - if the same node ID appears multiple times in the
/// graph, it will produce the same solution tree.
#[derive(Debug, Clone)]
struct MemoCache<L: Clone + Eq + Hash> {
    /// Cache for OR nodes: maps node ID -> (`best_solution_tree`, `min_distance`)
    or_cache: HashMap<OrId, MinEditResult<L>>,
    /// Cache for AND nodes: maps node ID -> (`solution_tree`, distance)
    and_cache: HashMap<AndId, MinEditResult<L>>,
}

impl<L: Clone + Eq + Hash> MemoCache<L> {
    fn new() -> Self {
        MemoCache {
            or_cache: HashMap::new(),
            and_cache: HashMap::new(),
        }
    }
}

/// Path tracker for cycle detection in AND-OR graphs.
/// Tracks how many times each OR node has been visited on the current path
/// and allows configurable revisit limits.
///
/// Only OR nodes are tracked because AND and OR nodes strictly alternate.
/// Cycles are detected at OR nodes since that's where paths can reconverge.
#[derive(Debug, Clone)]
struct PathTracker {
    /// Visit counts for OR nodes on the current path
    or_visits: HashMap<OrId, usize>,
    /// Maximum number of times any node may be revisited (0 = no revisits allowed)
    max_revisits: usize,
}

impl PathTracker {
    fn new(max_revisits: usize) -> Self {
        PathTracker {
            or_visits: HashMap::new(),
            max_revisits,
        }
    }

    /// Check if visiting this OR node would exceed the revisit limit.
    /// Returns true if the visit is allowed.
    fn can_visit(&self, id: OrId) -> bool {
        let count = self.or_visits.get(&id).copied().unwrap_or(0);
        count <= self.max_revisits
    }

    /// Mark an OR node as visited on the current path.
    fn enter(&mut self, id: OrId) {
        *self.or_visits.entry(id).or_insert(0) += 1;
    }

    /// Unmark an OR node when leaving the current path.
    fn leave(&mut self, id: OrId) {
        if let Some(count) = self.or_visits.get_mut(&id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.or_visits.remove(&id);
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
        // Graph: single AND leaf node "a", wrapped in an OR root
        let and_nodes = vec![AndNode::leaf("a")];
        let or_nodes = vec![OrNode::single("root", AndId(0))];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let solutions = graph.and(AndId(0)).generate_solutions(&graph);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].label(), &"a");
    }

    #[test]
    fn or_node_generates_multiple_trees() {
        // OR node with two AND leaf children -> two solution trees
        //   a(OR)
        //   / \
        //  b   c  (AND leaves)
        let and_nodes = vec![AndNode::leaf("b"), AndNode::leaf("c")];
        let or_nodes = vec![OrNode::new("a", vec![AndId(0), AndId(1)])];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let solutions = graph.or(OrId(0)).generate_solutions(&graph);
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

        let and_nodes = vec![
            AndNode::new("a", vec![OrId(0), OrId(1)]), // AndId(0)
            AndNode::leaf("d"),                        // AndId(1)
            AndNode::leaf("e"),                        // AndId(2)
            AndNode::leaf("f"),                        // AndId(3)
            AndNode::leaf("g"),                        // AndId(4)
        ];
        let or_nodes = vec![
            OrNode::new("b", vec![AndId(1), AndId(2)]), // OrId(0)
            OrNode::new("c", vec![AndId(3), AndId(4)]), // OrId(1)
            OrNode::single("root", AndId(0)),           // OrId(2) - root wrapper
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(2));

        let solutions = graph.and(AndId(0)).generate_solutions(&graph);
        assert_eq!(solutions.len(), 4);
        assert_eq!(graph.and(AndId(0)).count_solutions(&graph), 4);
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

        let and_nodes = vec![
            AndNode::new("root", vec![OrId(0)]), // AndId(0)
            AndNode::new("b", vec![OrId(1)]),    // AndId(1)
            AndNode::new("c", vec![OrId(2)]),    // AndId(2)
            AndNode::leaf("f"),                  // AndId(3)
            AndNode::leaf("g"),                  // AndId(4)
            AndNode::leaf("h"),                  // AndId(5)
        ];
        let or_nodes = vec![
            OrNode::new("a", vec![AndId(1), AndId(2)]), // OrId(0)
            OrNode::new("d", vec![AndId(3), AndId(4)]), // OrId(1)
            OrNode::single("e", AndId(5)),              // OrId(2)
            OrNode::single("graph_root", AndId(0)),     // OrId(3) - graph root wrapper
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(3));

        let solutions = graph.and(AndId(0)).generate_solutions(&graph);
        assert_eq!(solutions.len(), 3);
        assert_eq!(graph.and(AndId(0)).count_solutions(&graph), 3);
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

        let and_nodes = vec![AndNode::leaf("b"), AndNode::leaf("c")];
        let or_nodes = vec![OrNode::new("a", vec![AndId(0), AndId(1)])];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node("a", vec![leaf("b")]);

        let result = graph.find_min(&target, &UnitCost).unwrap();
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
    fn solution_count_complexity() {
        // Verify the counting matches actual enumeration
        //
        //           r (AND)
        //             |
        //           o1 (OR)
        //          /     \
        //       a1(AND)  a2(AND)
        //         |        |
        //       o2(OR)   o3(OR)
        //       / \       / \
        //      x   y     z   w

        let and_nodes = vec![
            AndNode::new("r", vec![OrId(0)]),  // AndId(0)
            AndNode::new("a1", vec![OrId(1)]), // AndId(1)
            AndNode::new("a2", vec![OrId(2)]), // AndId(2)
            AndNode::leaf("x"),                // AndId(3)
            AndNode::leaf("y"),                // AndId(4)
            AndNode::leaf("z"),                // AndId(5)
            AndNode::leaf("w"),                // AndId(6)
        ];
        let or_nodes = vec![
            OrNode::new("o1", vec![AndId(1), AndId(2)]), // OrId(0)
            OrNode::new("o2", vec![AndId(3), AndId(4)]), // OrId(1)
            OrNode::new("o3", vec![AndId(5), AndId(6)]), // OrId(2)
            OrNode::single("root", AndId(0)),            // OrId(3)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(3));

        let count = graph.and(AndId(0)).count_solutions(&graph);
        let solutions = graph.and(AndId(0)).generate_solutions(&graph);
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

        let and_nodes = vec![
            AndNode::new("root", vec![OrId(0), OrId(1)]), // AndId(0)
            AndNode::new("shared", vec![OrId(2)]),        // AndId(1) - shared!
            AndNode::leaf("x"),                           // AndId(2)
            AndNode::leaf("y"),                           // AndId(3)
        ];
        let or_nodes = vec![
            OrNode::single("o1", AndId(1)), // OrId(0) - points to shared
            OrNode::single("o2", AndId(1)), // OrId(1) - also points to shared (DAG!)
            OrNode::new("o3", vec![AndId(2), AndId(3)]), // OrId(2)
            OrNode::single("top", AndId(0)), // OrId(3) - graph root
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(3));

        // Target tree
        let target = node(
            "top",
            vec![node(
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
            )],
        );

        let result = graph.find_min_memo(&target, &UnitCost).unwrap();

        // The result should find a valid solution
        assert!(result.distance < usize::MAX);
    }

    #[test]
    fn dag_with_actual_shared_nodes() {
        // Create a true DAG where the same OR node is referenced by multiple AND nodes.
        // This tests that memoization correctly identifies shared structure.
        //
        //         root (OR)
        //        /        \
        //    and1 (AND)  and2 (AND)
        //       |    \    /    |
        //      o1    shared   o2
        //              |
        //         common (AND leaf)
        //
        // shared is the same OrNode referenced by both and1 and and2

        let and_nodes = vec![
            AndNode::new("and1", vec![OrId(0), OrId(1)]), // AndId(0)
            AndNode::new("and2", vec![OrId(1), OrId(2)]), // AndId(1) - also uses OrId(1)!
            AndNode::leaf("a"),                           // AndId(2)
            AndNode::leaf("common"),                      // AndId(3)
            AndNode::leaf("b"),                           // AndId(4)
        ];
        let or_nodes = vec![
            OrNode::single("o1", AndId(2)),                // OrId(0)
            OrNode::single("shared", AndId(3)),            // OrId(1) - shared by and1 and and2
            OrNode::single("o2", AndId(4)),                // OrId(2)
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(3)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(3));

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

        let result = graph.find_min(&target, &UnitCost).unwrap();

        // Should find and1 as the best match with distance 0
        assert_eq!(result.distance, 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_simple() {
        // Simple OR node with two choices
        let and_nodes = vec![AndNode::leaf("b"), AndNode::leaf("c")];
        let or_nodes = vec![OrNode::new("a", vec![AndId(0), AndId(1)])];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node("a", vec![leaf("b")]);

        let r1 = graph.find_min(&target, &UnitCost).unwrap();
        let r2 = graph.find_min_memo(&target, &UnitCost).unwrap();

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_nested() {
        // Nested structure: OR -> AND -> OR -> AND
        //
        //         root (OR)
        //           |
        //         a (AND)
        //        /      \
        //      b(OR)   c(OR)
        //      / \      / \
        //     d   e    f   g

        let and_nodes = vec![
            AndNode::new("a", vec![OrId(0), OrId(1)]), // AndId(0)
            AndNode::leaf("d"),                        // AndId(1)
            AndNode::leaf("e"),                        // AndId(2)
            AndNode::leaf("f"),                        // AndId(3)
            AndNode::leaf("g"),                        // AndId(4)
        ];
        let or_nodes = vec![
            OrNode::new("b", vec![AndId(1), AndId(2)]), // OrId(0)
            OrNode::new("c", vec![AndId(3), AndId(4)]), // OrId(1)
            OrNode::single("root", AndId(0)),           // OrId(2)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(2));

        let target = node(
            "root",
            vec![node(
                "a",
                vec![node("b", vec![leaf("d")]), node("c", vec![leaf("f")])],
            )],
        );

        let r1 = graph.find_min(&target, &UnitCost).unwrap();
        let r2 = graph.find_min_memo(&target, &UnitCost).unwrap();

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_multiple_choices() {
        // Multiple OR choices at different levels
        //
        //           root (OR)
        //          /        \
        //    choice1(AND)  choice2(AND)
        //        |            |
        //     inner(OR)    inner(OR)
        //      / \          / \
        //     x   y        z   w

        let and_nodes = vec![
            AndNode::new("choice1", vec![OrId(0)]), // AndId(0)
            AndNode::new("choice2", vec![OrId(1)]), // AndId(1)
            AndNode::leaf("x"),                     // AndId(2)
            AndNode::leaf("y"),                     // AndId(3)
            AndNode::leaf("z"),                     // AndId(4)
            AndNode::leaf("w"),                     // AndId(5)
        ];
        let or_nodes = vec![
            OrNode::new("inner", vec![AndId(2), AndId(3)]), // OrId(0)
            OrNode::new("inner", vec![AndId(4), AndId(5)]), // OrId(1)
            OrNode::new("root", vec![AndId(0), AndId(1)]),  // OrId(2)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(2));

        let target = node(
            "root",
            vec![node("choice1", vec![node("inner", vec![leaf("x")])])],
        );

        let r1 = graph.find_min(&target, &UnitCost).unwrap();
        let r2 = graph.find_min_memo(&target, &UnitCost).unwrap();

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn find_min_and_find_min_memo_equivalent_deep() {
        // Deeper alternation: OR -> AND -> OR -> AND -> OR -> AND(leaf)
        //
        //         l1 (OR)
        //           |
        //         l2 (AND)
        //           |
        //         l3 (OR)
        //        /      \
        //     l4a(AND)  l4b(AND)
        //       |         |
        //     l5(OR)    l5(OR)
        //     / \        / \
        //   leaf1 leaf2 leaf3 leaf4

        let and_nodes = vec![
            AndNode::new("l2", vec![OrId(0)]),  // AndId(0)
            AndNode::new("l4a", vec![OrId(1)]), // AndId(1)
            AndNode::new("l4b", vec![OrId(2)]), // AndId(2)
            AndNode::leaf("leaf1"),             // AndId(3)
            AndNode::leaf("leaf2"),             // AndId(4)
            AndNode::leaf("leaf3"),             // AndId(5)
            AndNode::leaf("leaf4"),             // AndId(6)
        ];
        let or_nodes = vec![
            OrNode::new("l3", vec![AndId(1), AndId(2)]), // OrId(0)
            OrNode::new("l5", vec![AndId(3), AndId(4)]), // OrId(1)
            OrNode::new("l5", vec![AndId(5), AndId(6)]), // OrId(2)
            OrNode::single("l1", AndId(0)),              // OrId(3)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(3));

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

        let r1 = graph.find_min(&target, &UnitCost).unwrap();
        let r2 = graph.find_min_memo(&target, &UnitCost).unwrap();

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    // ==================== Cycle handling tests ====================

    #[test]
    fn simple_cycle_blocked_with_no_revisits() {
        // Graph with a direct cycle: OR -> AND -> OR (same)
        //
        //   root (OR, id=0) <----+
        //        |               |
        //   loop_and (AND)       |
        //        |               |
        //   +----+---------------+
        //
        // This cycle has no escape, so with max_revisits=0 we get None

        let and_nodes = vec![
            AndNode::new("loop_and", vec![OrId(0)]), // AndId(0) - points back to root!
        ];
        let or_nodes = vec![
            OrNode::single("root", AndId(0)), // OrId(0) - the root that creates the cycle
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = leaf("anything");

        // With no revisits allowed, this should return None
        let result = graph.find_min_cyclic(&target, &UnitCost, 0);
        assert!(result.is_none());
    }

    #[test]
    fn cycle_with_escape_route() {
        // Graph with a cycle but also an escape route:
        //
        //       root (OR, id=0)
        //        /         \
        //   escape(AND)   loop_back(AND)
        //       |               |
        //      leaf         root (OR, id=0)  <- cycle!
        //
        // With max_revisits=0, should find the escape route

        let and_nodes = vec![
            AndNode::leaf("escape"),             // AndId(0) - escape route
            AndNode::new("loop", vec![OrId(0)]), // AndId(1) - creates cycle
        ];
        let or_nodes = vec![
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(0)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node("root", vec![leaf("escape")]);

        let result = graph.find_min_cyclic(&target, &UnitCost, 0).unwrap();

        assert_eq!(result.distance, 0);
    }

    #[test]
    fn cycle_with_revisit_allows_unrolling() {
        // Graph where revisiting produces a valid tree:
        //
        //       root (OR, id=0)
        //        /         \
        //   recurse(AND)   base(AND)
        //       |              |
        //   inner(OR)        leaf
        //       |
        //     root (OR, id=0)  <- back to root
        //
        // With max_revisits=1, we can unroll once

        let and_nodes = vec![
            AndNode::new("recurse", vec![OrId(1)]), // AndId(0)
            AndNode::leaf("base"),                  // AndId(1)
        ];
        let or_nodes = vec![
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(0)
            OrNode::new("inner", vec![AndId(0), AndId(1)]), // OrId(1) - choices including recursive
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = leaf("base");

        // With 0 revisits, should still find base
        let result = graph.find_min_cyclic(&target, &UnitCost, 0);
        assert!(result.is_some());
    }

    #[test]
    fn deep_cycle_respects_revisit_limit() {
        // A longer cycle: a -> b -> c -> a
        //
        //     a (OR)
        //       |
        //    and_a (AND)
        //       |
        //     b (OR)
        //       |
        //    and_b (AND)
        //       |
        //     c (OR)
        //      / \
        // and_c   leaf
        //    |
        //    a (OR)  <- cycle back

        let and_nodes = vec![
            AndNode::new("and_a", vec![OrId(1)]), // AndId(0)
            AndNode::new("and_b", vec![OrId(2)]), // AndId(1)
            AndNode::new("and_c", vec![OrId(0)]), // AndId(2) - cycles back to a!
            AndNode::leaf("leaf"),                // AndId(3) - escape
        ];
        let or_nodes = vec![
            OrNode::single("a", AndId(0)),              // OrId(0)
            OrNode::single("b", AndId(1)),              // OrId(1)
            OrNode::new("c", vec![AndId(2), AndId(3)]), // OrId(2) - has escape
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = leaf("anything");

        // With 0 revisits, should find the escape
        // Result should use the leaf escape route
        let result = graph.find_min_cyclic(&target, &UnitCost, 0).unwrap();

        assert!(result.tree.label() == &"a");
    }

    #[test]
    fn cyclic_and_memo_cyclic_equivalent() {
        // Test that find_min_cyclic and find_min_memo_cyclic produce equivalent results
        // on a graph with cycles

        let and_nodes = vec![
            AndNode::new("recurse", vec![OrId(1)]), // AndId(0)
            AndNode::leaf("base"),                  // AndId(1)
        ];
        let or_nodes = vec![
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(0)
            OrNode::new("inner", vec![AndId(0), AndId(1)]), // OrId(1)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node("root", vec![leaf("base")]);

        let r1 = graph.find_min_cyclic(&target, &UnitCost, 0).unwrap();
        let r2 = graph.find_min_memo_cyclic(&target, &UnitCost, 0).unwrap();

        assert_eq!(r1.distance, r2.distance);
        assert_eq!(tree_distance_unit(&r1.tree, &r2.tree), 0);
    }

    #[test]
    fn no_cycle_graph_works_with_cyclic_methods() {
        // Test that graphs without cycles work correctly with the cyclic methods
        let and_nodes = vec![AndNode::leaf("b"), AndNode::leaf("c")];
        let or_nodes = vec![OrNode::new("a", vec![AndId(0), AndId(1)])];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node("a", vec![leaf("b")]);

        let r_normal = graph.find_min(&target, &UnitCost).unwrap();
        let r_memo = graph.find_min_memo(&target, &UnitCost).unwrap();

        assert_eq!(r_normal.distance, r_memo.distance);
    }

    #[test]
    fn revisit_limit_allows_controlled_unrolling() {
        // Graph where higher revisit limits allow deeper unrolling:
        //
        //        root (OR)
        //         /     \
        //    recurse    base (AND leaf)
        //    (AND)
        //       |
        //     root (OR)  <- cycles back
        //
        // With max_revisits=0: can only take base path
        // With max_revisits=1: can unroll once (recurse -> base)

        let and_nodes = vec![
            AndNode::new("recurse", vec![OrId(0)]), // AndId(0) - cycles back to root
            AndNode::leaf("base"),                  // AndId(1) - escape
        ];
        let or_nodes = vec![
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(0) - choice of recurse or base
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = leaf("anything");

        // With 0 revisits, should find the base escape immediately
        let r_0 = graph.find_min_cyclic(&target, &UnitCost, 0).unwrap();
        // The result uses the base path (shortest)
        assert_eq!(r_0.tree.children().len(), 1);
        assert_eq!(r_0.tree.children()[0].label(), &"base");

        // With 1 revisit, we can unroll once and choose either path
        let _ = graph.find_min_cyclic(&target, &UnitCost, 1).unwrap();
    }

    #[test]
    fn multiple_cycles_handled_independently() {
        // Graph with two separate cycles that can be navigated
        //
        //         root (OR)
        //          /     \
        //       a (AND)  b (AND)
        //        |         |
        //     o_a (OR)   o_b (OR)
        //      /  \       /   \
        //  leaf   root  leaf  root
        //
        // Both branches cycle back but have escapes

        let and_nodes = vec![
            AndNode::new("a", vec![OrId(1)]),       // AndId(0)
            AndNode::new("b", vec![OrId(2)]),       // AndId(1)
            AndNode::leaf("leaf_a"),                // AndId(2)
            AndNode::leaf("leaf_b"),                // AndId(3)
            AndNode::new("cycle_a", vec![OrId(0)]), // AndId(4) - cycles to root
            AndNode::new("cycle_b", vec![OrId(0)]), // AndId(5) - cycles to root
        ];
        let or_nodes = vec![
            OrNode::new("root", vec![AndId(0), AndId(1)]), // OrId(0)
            OrNode::new("o_a", vec![AndId(2), AndId(4)]),  // OrId(1)
            OrNode::new("o_b", vec![AndId(3), AndId(5)]),  // OrId(2)
        ];
        let graph = AndOrGraph::new(or_nodes, and_nodes, OrId(0));

        let target = node(
            "root",
            vec![node("a", vec![node("o_a", vec![leaf("leaf_a")])])],
        );

        let result = graph.find_min_cyclic(&target, &UnitCost, 0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().distance, 0);
    }
}
