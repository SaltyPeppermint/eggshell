//! `EGraph` Extension for Zhang-Shasha Tree Edit Distance
//!
//! Finds the solution tree in a bounded `EGraph` with minimum edit distance
//! to a target tree. Assumes bounded maximum number of nodes in an `EClass` (N) and bounded depth (d).
//!
//! With strict alternation (`EClass` -> `ENode` -> `EClass` ->...),
//! complexity is O(N^(d/2) * |T|^2) for single-path graphs

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::TreeNode;
use super::ids::{DataTyId, EClassId, FunTyId, NatId, TypeId};
use super::nodes::{DataTyNode, ENode, FunTyNode, Label, NatNode};
use super::tree::{EditCosts, UnitCost, tree_distance};

/// `EClass`: choose exactly one child (`ENode`)
/// Children are `ENode` instances directly
/// Must have at least one child
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EClass<L: Label> {
    children: Vec<ENode<L>>,
    ty: TypeId,
}

impl<L: Label> EClass<L> {
    #[must_use]
    pub fn new(children: Vec<ENode<L>>, ty: TypeId) -> Self {
        Self { children, ty }
    }

    #[must_use]
    pub fn children(&self) -> &[ENode<L>] {
        &self.children
    }

    #[must_use]
    pub fn ty(&self) -> TypeId {
        self.ty
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EGraph<L: Label> {
    classes: Vec<EClass<L>>,
    root: EClassId,
    fun_ty_nodes: HashMap<FunTyId, FunTyNode<L>>,
    nat_nodes: HashMap<NatId, NatNode<L>>,
    data_ty_nodes: HashMap<DataTyId, DataTyNode<L>>,
}

impl<L: Label> EGraph<L> {
    #[must_use]
    pub fn new(
        classes: Vec<EClass<L>>,
        root: EClassId,
        type_nodes: HashMap<FunTyId, FunTyNode<L>>,
        nat_nodes: HashMap<NatId, NatNode<L>>,
        data_type_nodes: HashMap<DataTyId, DataTyNode<L>>,
    ) -> Self {
        Self {
            classes,
            root,
            fun_ty_nodes: type_nodes,
            nat_nodes,
            data_ty_nodes: data_type_nodes,
        }
    }

    #[must_use]
    pub fn class(&self, id: EClassId) -> &EClass<L> {
        &self.classes[usize::from(id)]
    }

    #[must_use]
    pub fn fun_ty(&self, id: FunTyId) -> &FunTyNode<L> {
        &self.fun_ty_nodes[&id]
    }

    #[must_use]
    pub fn data_ty(&self, id: DataTyId) -> &DataTyNode<L> {
        &self.data_ty_nodes[&id]
    }

    #[must_use]
    pub fn nat(&self, id: NatId) -> &NatNode<L> {
        &self.nat_nodes[&id]
    }

    /// Enumerate all possible trees extractable from an `EGraph` starting at a given class,
    /// respecting the cycle revisit limit.
    ///
    /// Returns the tree and the index of the last choice used, or None if no more trees exist.
    /// The `choices` vector is modified to record/follow choices at each `EClass`.
    fn get_next_tree(
        &self,
        id: EClassId,
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
        for (node_idx, node) in class.children().iter().enumerate().skip(choice) {
            // Set the choice_idx for future choices
            // Useless write if the choice is correct
            choices[choice_idx] = node_idx;

            let result = node.children().iter().try_fold(
                (Vec::new(), choice_idx),
                |(mut children, curr_idx): (Vec<_>, _), child_id| {
                    let (child_tree, last_idx) =
                        self.get_next_tree(*child_id, curr_idx + 1, choices, path)?;
                    children.push(child_tree);
                    Some((children, last_idx))
                },
            );

            if let Some((children, curr_idx)) = result {
                path.leave(id);
                let expr_tree = TreeNode::new(node.label().clone(), children);
                let type_tree = TreeNode::from_eclass(self, id);
                let eclass_tree = TreeNode::new(L::type_of(), vec![expr_tree, type_tree]);
                return Some((eclass_tree, curr_idx));
            }

            // This node's children failed, try next node in this class
            // Reset choices for children (truncate to current position + 1)
            choices.truncate(choice_idx + 1);
        }

        path.leave(id);
        // No valid choice found at this level, need to backtrack
        None
    }

    #[must_use]
    pub fn tree_from_choices(&self, id: EClassId, choices: &[usize]) -> TreeNode<L> {
        fn rec<L: Label>(
            graph: &EGraph<L>,
            id: EClassId,
            choice_idx: usize,
            choices: &[usize],
        ) -> (TreeNode<L>, usize) {
            let class = graph.class(id);
            let choice = choices[choice_idx];
            let node = &class.children[choice];

            let (children, curr_idx) = node.children().iter().fold(
                (Vec::new(), choice_idx),
                |(mut children, curr_idx): (Vec<_>, _), child_id| {
                    let (child_tree, last_idx) = rec(graph, *child_id, curr_idx + 1, choices);
                    children.push(child_tree);
                    (children, last_idx)
                },
            );

            let expr_tree = TreeNode::new(node.label().clone(), children);
            let type_tree = TreeNode::from_eclass(graph, id);
            let eclass_tree = TreeNode::new(L::type_of(), vec![expr_tree, type_tree]);
            (eclass_tree, curr_idx)
        }
        rec(self, id, 0, choices).0
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

#[derive(Debug)]
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

/// Result of finding the minimum edit distance solution tree
#[derive(Debug, Clone)]
pub struct MinEditResult<L: Label> {
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

    fn leaf<L: Label>(label: impl Into<L>) -> TreeNode<L> {
        TreeNode::leaf(label.into())
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::new(label, children)
    }

    fn eid(i: usize) -> EClassId {
        EClassId::new(i)
    }

    /// Helper to create a dummy `TypeId` for tests
    fn dummy_ty() -> TypeId {
        TypeId::Nat(NatId::new(0))
    }

    /// Helper to create dummy nat nodes hashmap with a "0" leaf at NatId(0)
    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut nats = HashMap::new();
        nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        nats
    }

    /// Helper to build a simple graph with one class containing one node
    fn single_node_graph(label: &str) -> EGraph<String> {
        EGraph::new(
            vec![EClass::new(vec![ENode::leaf(label.to_owned())], dummy_ty())],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        )
    }

    #[test]
    fn enumerate_single_leaf() {
        let graph = single_node_graph("a");
        let trees = graph.enumerate_trees(0);

        assert_eq!(trees.len(), 1);
        // Trees are now wrapped: typeOf(expr_tree, type_tree)
        assert_eq!(trees[0].label(), "typeOf");
        assert_eq!(trees[0].children().len(), 2);
        assert_eq!(trees[0].children()[0].label(), "a");
    }

    #[test]
    fn enumerate_with_or_choice() {
        // Graph with one class containing two node choices
        let graph = EGraph::new(
            vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 2);

        // Extract expr labels from typeOf(expr, type) wrapper
        let labels: Vec<_> = trees
            .iter()
            .map(|t| t.children()[0].label().as_str())
            .collect();
        assert!(labels.contains(&"a"));
        assert!(labels.contains(&"b"));
    }

    #[test]
    fn enumerate_with_and_children() {
        // Graph: root class -> node with two child classes (each has one leaf)
        // Class 0: root, has node "a" pointing to classes 1 and 2
        // Class 1: leaf "b"
        // Class 2: leaf "c"
        let graph = EGraph::new(
            vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 1);
        // Root is typeOf(expr, type), expr is "a" with children that are also typeOf wrapped
        let expr = &trees[0].children()[0];
        assert_eq!(expr.label(), "a");
        assert_eq!(expr.children().len(), 2);
        // Children are typeOf(b, type) and typeOf(c, type)
        assert_eq!(expr.children()[0].children()[0].label(), "b");
        assert_eq!(expr.children()[1].children()[0].label(), "c");
    }

    #[test]
    fn enumerate_with_cycle_no_revisits() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            vec![EClass::new(
                vec![
                    ENode::new("a".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 0 revisits, we can only take the leaf option
        let trees = graph.enumerate_trees(0);
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].children()[0].label(), "leaf");
    }

    #[test]
    fn enumerate_with_cycle_one_revisit() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            vec![EClass::new(
                vec![
                    ENode::new("rec".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 1 revisit, we can go one level deep
        let trees = graph.enumerate_trees(1);

        // Should have: "leaf", "rec(leaf)", "rec(rec(leaf))"
        // Actually: at depth 0 we have 2 choices, at depth 1 we have 2 choices...
        // Let's verify we get more trees than with 0 revisits
        assert!(trees.len() > 1);

        // Check that we have the recursive structure (expr part of typeOf wrapper)
        let has_recursive = trees
            .iter()
            .any(|t| t.children()[0].label() == "rec" && !t.children()[0].children().is_empty());
        assert!(has_recursive);
    }

    #[test]
    fn min_distance_exact_match() {
        // Graph contains the exact reference tree
        let graph = EGraph::new(
            vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Reference must match the typeOf wrapper structure
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
                leaf("0".to_owned()), // type tree (dummy nat)
            ],
        );
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
    }

    #[test]
    fn min_distance_chooses_best() {
        // Graph with OR choice: "a" or "x"
        // Reference is "a", so should choose "a" with distance 0
        let graph = EGraph::new(
            vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("x".to_owned())],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Reference must match typeOf wrapper
        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
        assert_eq!(result.tree.children()[0].label(), "a");
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
                EClass::new(
                    vec![
                        ENode::new("a".to_owned(), vec![eid(1)]),         // a(b)
                        ENode::new("a".to_owned(), vec![eid(1), eid(2)]), // a(b, c)
                    ],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Reference with typeOf wrapper: typeOf(a(typeOf(b, 0)), 0)
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
        let result = min_distance_extract_unit(&graph, &reference, 0).unwrap();

        assert_eq!(result.distance, 0);
        // The expr part of typeOf wrapper should have 1 child
        assert_eq!(result.tree.children()[0].children().len(), 1);
    }

    #[test]
    fn tree_from_choices_single_leaf() {
        let graph = single_node_graph("a");
        let choices = vec![0];

        let tree = graph.tree_from_choices(eid(0), &choices);

        // typeOf wrapper
        assert_eq!(tree.label(), "typeOf");
        assert_eq!(tree.children()[0].label(), "a");
    }

    #[test]
    fn tree_from_choices_or_choice_first() {
        // Graph with two OR choices
        let graph = EGraph::new(
            vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0];
        let tree = graph.tree_from_choices(eid(0), &choices);

        assert_eq!(tree.children()[0].label(), "a");
    }

    #[test]
    fn tree_from_choices_or_choice_second() {
        // Graph with two OR choices
        let graph = EGraph::new(
            vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![1];
        let tree = graph.tree_from_choices(eid(0), &choices);

        assert_eq!(tree.children()[0].label(), "b");
    }

    #[test]
    fn tree_from_choices_with_and_children() {
        // Graph: root -> node with two child classes
        let graph = EGraph::new(
            vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0, 0, 0];
        let tree = graph.tree_from_choices(eid(0), &choices);

        // typeOf(a(...), type)
        let expr = &tree.children()[0];
        assert_eq!(expr.label(), "a");
        assert_eq!(expr.children().len(), 2);
        // Children are typeOf wrapped
        assert_eq!(expr.children()[0].children()[0].label(), "b");
        assert_eq!(expr.children()[1].children()[0].label(), "c");
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
                EClass::new(
                    vec![
                        ENode::new("x".to_owned(), vec![eid(1)]),
                        ENode::new("y".to_owned(), vec![eid(1)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test x(a) - expr part of wrapper
        let choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(eid(0), &choices1);
        assert_eq!(tree1.children()[0].label(), "x");
        assert_eq!(tree1.children()[0].children()[0].children()[0].label(), "a");

        // Test x(b)
        let choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(eid(0), &choices2);
        assert_eq!(tree2.children()[0].label(), "x");
        assert_eq!(tree2.children()[0].children()[0].children()[0].label(), "b");

        // Test y(a)
        let choices3 = vec![1, 0];
        let tree3 = graph.tree_from_choices(eid(0), &choices3);
        assert_eq!(tree3.children()[0].label(), "y");
        assert_eq!(tree3.children()[0].children()[0].children()[0].label(), "a");

        // Test y(b)
        let choices4 = vec![1, 1];
        let tree4 = graph.tree_from_choices(eid(0), &choices4);
        assert_eq!(tree4.children()[0].label(), "y");
        assert_eq!(tree4.children()[0].children()[0].children()[0].label(), "b");
    }

    #[test]
    fn tree_from_choices_multiple_and_children_with_or_choices() {
        // Graph with multiple AND children each having OR choices:
        // Class 0: root -> "p" with children [Class 1, Class 2]
        // Class 1: "a" or "b"
        // Class 2: "x" or "y"
        let graph = EGraph::new(
            vec![
                EClass::new(
                    vec![ENode::new("p".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("x".to_owned()), ENode::leaf("y".to_owned())],
                    dummy_ty(),
                ),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test p(a, x) - check expr children (which are typeOf wrapped)
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(eid(0), &choices1);
        let expr1 = &tree1.children()[0];
        assert_eq!(expr1.label(), "p");
        assert_eq!(expr1.children()[0].children()[0].label(), "a");
        assert_eq!(expr1.children()[1].children()[0].label(), "x");

        // Test p(a, y)
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(eid(0), &choices2);
        let expr2 = &tree2.children()[0];
        assert_eq!(expr2.label(), "p");
        assert_eq!(expr2.children()[0].children()[0].label(), "a");
        assert_eq!(expr2.children()[1].children()[0].label(), "y");

        // Test p(b, x)
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(eid(0), &choices3);
        let expr3 = &tree3.children()[0];
        assert_eq!(expr3.label(), "p");
        assert_eq!(expr3.children()[0].children()[0].label(), "b");
        assert_eq!(expr3.children()[1].children()[0].label(), "x");

        // Test p(b, y)
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(eid(0), &choices4);
        let expr4 = &tree4.children()[0];
        assert_eq!(expr4.label(), "p");
        assert_eq!(expr4.children()[0].children()[0].label(), "b");
        assert_eq!(expr4.children()[1].children()[0].label(), "y");
    }

    #[test]
    fn tree_from_choices_deep_nesting() {
        // Deep tree with choices at multiple levels
        // Class 0: root -> "a" with child [Class 1]
        // Class 1: "b1" or "b2", both with child [Class 2]
        // Class 2: "c1" or "c2"
        let graph = EGraph::new(
            vec![
                EClass::new(vec![ENode::new("a".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![
                        ENode::new("b1".to_owned(), vec![eid(2)]),
                        ENode::new("b2".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("c1".to_owned()), ENode::leaf("c2".to_owned())],
                    dummy_ty(),
                ),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test a(b1(c1)) - navigate through typeOf wrappers
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(eid(0), &choices1);
        let expr1 = &tree1.children()[0]; // a
        assert_eq!(expr1.label(), "a");
        let b1 = &expr1.children()[0].children()[0]; // typeOf -> b1
        assert_eq!(b1.label(), "b1");
        let c1 = &b1.children()[0].children()[0]; // typeOf -> c1
        assert_eq!(c1.label(), "c1");

        // Test a(b1(c2))
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(eid(0), &choices2);
        let expr2 = &tree2.children()[0];
        let b1_2 = &expr2.children()[0].children()[0];
        let c2 = &b1_2.children()[0].children()[0];
        assert_eq!(c2.label(), "c2");

        // Test a(b2(c1))
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(eid(0), &choices3);
        let expr3 = &tree3.children()[0];
        let b2 = &expr3.children()[0].children()[0];
        assert_eq!(b2.label(), "b2");
        let c1_3 = &b2.children()[0].children()[0];
        assert_eq!(c1_3.label(), "c1");

        // Test a(b2(c2))
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(eid(0), &choices4);
        let expr4 = &tree4.children()[0];
        let b2_4 = &expr4.children()[0].children()[0];
        assert_eq!(b2_4.label(), "b2");
        let c2_4 = &b2_4.children()[0].children()[0];
        assert_eq!(c2_4.label(), "c2");
    }

    #[test]
    fn tree_from_choices_three_and_children() {
        // Test with three AND children
        let graph = EGraph::new(
            vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2), eid(3)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0, 0, 0, 0];
        let tree = graph.tree_from_choices(eid(0), &choices);

        let expr = &tree.children()[0];
        assert_eq!(expr.label(), "f");
        assert_eq!(expr.children().len(), 3);
        // Children are typeOf wrapped
        assert_eq!(expr.children()[0].children()[0].label(), "a");
        assert_eq!(expr.children()[1].children()[0].label(), "b");
        assert_eq!(expr.children()[2].children()[0].label(), "c");
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
                EClass::new(
                    vec![
                        ENode::new("x".to_owned(), vec![eid(1)]),
                        ENode::leaf("y".to_owned()),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ],
            eid(0),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let enumerated = graph.enumerate_trees(0);

        // Should produce: x(a), x(b), y
        assert_eq!(enumerated.len(), 3);

        // Reconstruct using tree_from_choices
        let choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(eid(0), &choices1);
        assert!(trees_equal(&tree1, &enumerated[0]));

        let choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(eid(0), &choices2);
        assert!(trees_equal(&tree2, &enumerated[1]));

        let choices3 = vec![1];
        let tree3 = graph.tree_from_choices(eid(0), &choices3);
        assert!(trees_equal(&tree3, &enumerated[2]));
    }
}
