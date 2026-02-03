//! `EGraph` Extension for Zhang-Shasha Tree Edit Distance
//!
//! Finds the solution tree in a bounded `EGraph` with minimum edit distance
//! to a target tree. Assumes bounded maximum number of nodes in an `EClass` (N) and bounded depth (d).
//!
//! With strict alternation (`EClass` -> `ENode` -> `EClass` ->...),
//! complexity is O(N^(d/2) * |T|^2) for single-path graphs

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use indicatif::ParallelProgressIterator;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::distance::structural::structural_diff;

use super::ids::{
    DataId, EClassId, ExprChildId, FunId, NatId, NumericId, TypeChildId, eclass_id_vec,
    numeric_key_map,
};
use super::nodes::{DataTyNode, ENode, FunTyNode, Label, NatNode};
use super::str::EulerString;
use super::tree::TreeNode;
use super::zs::{EditCosts, PreprocessedTree, tree_distance_with_ref};

/// `EClass`: choose exactly one child (`ENode`)
/// Children are `ENode` instances directly
/// Must have at least one child
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EClass<L: Label> {
    nodes: Vec<ENode<L>>,
    ty: TypeChildId,
}

impl<L: Label> EClass<L> {
    #[must_use]
    pub fn new(nodes: Vec<ENode<L>>, ty: TypeChildId) -> Self {
        Self { nodes, ty }
    }

    #[must_use]
    pub fn nodes(&self) -> &[ENode<L>] {
        &self.nodes
    }

    #[must_use]
    pub fn ty(&self) -> TypeChildId {
        self.ty
    }
}

/// E-graph with type annotations for tree extraction and edit distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EGraph<L: Label> {
    #[serde(with = "numeric_key_map")]
    classes: HashMap<EClassId, EClass<L>>,
    #[serde(skip)]
    root: Option<EClassId>,
    #[serde(rename = "unionFind", with = "eclass_id_vec")]
    union_find: Vec<EClassId>,
    #[serde(rename = "typeHashCon", with = "numeric_key_map")]
    fun_ty_nodes: HashMap<FunId, FunTyNode<L>>,
    #[serde(rename = "natHashCon", with = "numeric_key_map")]
    nat_nodes: HashMap<NatId, NatNode<L>>,
    #[serde(rename = "dataTypeHashCon", with = "numeric_key_map")]
    data_ty_nodes: HashMap<DataId, DataTyNode<L>>,
}

impl<L: Label> EGraph<L> {
    #[must_use]
    pub fn new(
        classes: HashMap<EClassId, EClass<L>>,
        root: EClassId,
        union_find: Vec<EClassId>,
        fun_ty_nodes: HashMap<FunId, FunTyNode<L>>,
        nat_nodes: HashMap<NatId, NatNode<L>>,
        data_ty_nodes: HashMap<DataId, DataTyNode<L>>,
    ) -> Self {
        Self {
            classes,
            root: Some(root),
            union_find,
            fun_ty_nodes,
            nat_nodes,
            data_ty_nodes,
        }
    }

    /// Set the root `EClassId` (useful after deserialization)
    fn set_root(&mut self, root: EClassId) {
        self.root = Some(root);
    }

    /// Get the root `EClassId`.
    ///
    /// # Panics
    /// Panics if the root has not been set.
    #[must_use]
    pub fn root(&self) -> EClassId {
        self.root
            .expect("Root has not been set. This is necessary after deserializing the egraph!")
    }

    /// Parse an `EGraph` from a JSON file, extracting the root ID from the filename.
    ///
    /// Expects filename format: `..._root_<id>.json` (e.g., `ser_egraph_root_129.json`).
    ///
    /// # Panics
    /// Panics if the file cannot be read, parsed, or if the filename doesn't match the expected format.
    #[must_use]
    pub fn parse_from_file(file: &Path) -> EGraph<L> {
        let mut graph: EGraph<L> =
            serde_json::from_reader(BufReader::new(File::open(file).unwrap())).unwrap();
        // Pattern: ..._root_123.json
        let stem = file.file_stem().unwrap().to_str().unwrap();
        let id_str = stem.split('_').next_back().unwrap();
        graph.set_root(EClassId::new(id_str.parse().unwrap()));
        graph
    }

    /// Canonicalize an `EClassId` through the union-find.
    #[must_use]
    pub fn canonicalize(&self, id: EClassId) -> EClassId {
        if self.union_find.is_empty() {
            return id;
        }
        let mut current = id;
        while self.union_find[current.to_index()] != current {
            current = self.union_find[current.to_index()];
        }
        current
    }

    #[must_use]
    /// Returns the corresponding `EClass`. Can take a non-canonical Id
    pub fn class(&self, id: EClassId) -> &EClass<L> {
        let canonical = self.canonicalize(id);
        &self.classes[&canonical]
    }

    #[must_use]
    pub fn fun_ty(&self, id: FunId) -> &FunTyNode<L> {
        &self.fun_ty_nodes[&id]
    }

    #[must_use]
    pub fn data_ty(&self, id: DataId) -> &DataTyNode<L> {
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
    fn next_tree(
        &self,
        id: EClassId,
        choice_idx: usize,
        choices: &mut Vec<usize>,
        path: &mut PathTracker,
        with_type: bool,
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
        for (node_idx, node) in class.nodes().iter().enumerate().skip(choice) {
            // Set the choice_idx for future choices
            // Useless write if the choice is correct
            choices[choice_idx] = node_idx;

            let result = node.children().iter().try_fold(
                (Vec::new(), choice_idx),
                |(mut children, curr_idx): (Vec<_>, _), child_id| {
                    let (child_tree, last_idx) = match child_id {
                        // With nats and dt we cannot make a choice so do not add anything to the choice
                        ExprChildId::Nat(nat_id) => (TreeNode::from_nat(self, *nat_id), curr_idx),
                        ExprChildId::Data(dt_id) => (TreeNode::from_data(self, *dt_id), curr_idx),
                        ExprChildId::EClass(eclass_id) => {
                            self.next_tree(*eclass_id, curr_idx + 1, choices, path, with_type)?
                        }
                    };

                    children.push(child_tree);
                    Some((children, last_idx))
                },
            );

            if let Some((children, curr_idx)) = result {
                path.leave(id);
                let expr_tree = TreeNode::new(node.label().clone(), children);
                let eclass_tree = if with_type {
                    let type_tree = TreeNode::from_eclass(self, id);
                    TreeNode::new(L::type_of(), vec![expr_tree, type_tree])
                } else {
                    expr_tree
                };
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

    /// Advance to the next valid choice vector without materializing the tree.
    /// Returns the last choice index used, or None if no more valid choices exist.
    fn next_choices(
        &self,
        id: EClassId,
        choice_idx: usize,
        choices: &mut Vec<usize>,
        path: &mut PathTracker,
    ) -> Option<usize> {
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
        for (node_idx, node) in class.nodes().iter().enumerate().skip(choice) {
            choices[choice_idx] = node_idx;

            let result = node
                .children()
                .iter()
                .try_fold(choice_idx, |curr_idx, child_id| match child_id {
                    // With nats and dt we cannot make a choice so do not add anything to the choice
                    ExprChildId::Nat(_) | ExprChildId::Data(_) => Some(curr_idx),
                    ExprChildId::EClass(eclass_id) => {
                        self.next_choices(*eclass_id, curr_idx + 1, choices, path)
                    }
                });

            if let Some(curr_idx) = result {
                path.leave(id);
                return Some(curr_idx);
            }

            // This node's children failed, try next node in this class
            choices.truncate(choice_idx + 1);
        }

        path.leave(id);
        None
    }

    #[must_use]
    pub fn tree_from_choices(
        &self,
        id: EClassId,
        choices: &[usize],
        with_type: bool,
    ) -> TreeNode<L> {
        self.tree_from_choices_rec(id, 0, choices, with_type).0
    }

    fn tree_from_choices_rec(
        &self,
        id: EClassId,
        choice_idx: usize,
        choices: &[usize],
        with_type: bool,
    ) -> (TreeNode<L>, usize) {
        let class = self.class(id);
        let choice = choices[choice_idx];
        let node = &class.nodes()[choice];

        let (children, curr_idx) = node.children().iter().fold(
            (Vec::new(), choice_idx),
            |(mut children, curr_idx): (Vec<_>, _), child_id| {
                let (child_tree, last_idx) = match child_id {
                    // With nats and dt we cannot make a choice so do not add anything to the choice
                    ExprChildId::Nat(nat_id) => (TreeNode::from_nat(self, *nat_id), curr_idx),
                    ExprChildId::Data(dt_id) => (TreeNode::from_data(self, *dt_id), curr_idx),
                    ExprChildId::EClass(eclass_id) => {
                        self.tree_from_choices_rec(*eclass_id, curr_idx + 1, choices, with_type)
                    }
                };
                children.push(child_tree);
                (children, last_idx)
            },
        );

        let expr_tree = TreeNode::new(node.label().clone(), children);
        let eclass_tree = if with_type {
            let type_tree = TreeNode::from_eclass(self, id);
            TreeNode::new(L::type_of(), vec![expr_tree, type_tree])
        } else {
            expr_tree
        };
        (eclass_tree, curr_idx)
    }

    #[must_use]
    pub fn enumerate_trees(&self, max_revisits: usize, with_type: bool) -> Vec<TreeNode<L>> {
        TreeIter::new(self, max_revisits, with_type).collect()
    }

    #[must_use]
    pub fn count_trees(&self, max_revisits: usize) -> usize {
        ChoiceIter::new(self, max_revisits).count()
    }

    #[must_use]
    pub fn choice_iter(&self, max_revisits: usize) -> ChoiceIter<'_, L> {
        ChoiceIter::new(self, max_revisits)
    }

    #[must_use]
    pub fn tree_iter(&self, max_revisits: usize, with_type: bool) -> TreeIter<'_, L> {
        TreeIter::new(self, max_revisits, with_type)
    }
}

/// If `quiet` is true, hides the progress bar.
/// If `strip_types`, ignores the types
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

/// If `quiet` is true, hides the progress bar.
/// If `strip_types`, ignores the types
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

#[derive(Debug)]
pub struct TreeIter<'a, L: Label> {
    choices: Vec<usize>,
    path: PathTracker,
    egraph: &'a EGraph<L>,
    with_type: bool,
}

impl<'a, L: Label> TreeIter<'a, L> {
    pub fn new(egraph: &'a EGraph<L>, max_revisits: usize, with_type: bool) -> Self {
        Self {
            choices: Vec::new(),
            path: PathTracker::new(max_revisits),
            egraph,
            with_type,
        }
    }
}

impl<L: Label> Iterator for TreeIter<'_, L> {
    type Item = TreeNode<L>;

    fn next(&mut self) -> Option<Self::Item> {
        let (tree, _) = self.egraph.next_tree(
            self.egraph.root(),
            0,
            &mut self.choices,
            &mut self.path,
            self.with_type,
        )?;
        if let Some(last) = self.choices.last_mut() {
            *last += 1;
        }
        Some(tree)
    }
}

/// Iterator that yields choice vectors without materializing trees.
/// Each choice vector can later be used with `tree_from_choices` to get the actual tree.
#[derive(Debug)]
pub struct ChoiceIter<'a, L: Label> {
    choices: Vec<usize>,
    path: PathTracker,
    egraph: &'a EGraph<L>,
}

impl<'a, L: Label> ChoiceIter<'a, L> {
    pub fn new(egraph: &'a EGraph<L>, max_revisits: usize) -> Self {
        Self {
            choices: Vec::new(),
            path: PathTracker::new(max_revisits),
            egraph,
        }
    }
}

impl<L: Label> Iterator for ChoiceIter<'_, L> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.egraph
            .next_choices(self.egraph.root(), 0, &mut self.choices, &mut self.path)?;
        let result = self.choices.clone();
        if let Some(last) = self.choices.last_mut() {
            *last += 1;
        }
        Some(result)
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
    fn size_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 1,
            euler_pruned: 0,
            full_comparisons: 0,
        }
    }

    fn euler_pruned() -> Self {
        Self {
            trees_enumerated: 1,
            size_pruned: 0,
            euler_pruned: 1,
            full_comparisons: 0,
        }
    }

    fn compared() -> Self {
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

#[cfg(test)]
mod tests {

    use crate::distance::ids::NumericId;
    use crate::distance::zs::UnitCost;

    use super::*;

    fn leaf<L: Label>(label: impl Into<L>) -> TreeNode<L> {
        TreeNode::leaf(label.into())
    }

    fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
        TreeNode::new(label, children)
    }

    fn eid(i: usize) -> ExprChildId {
        ExprChildId::EClass(EClassId::new(i))
    }

    /// Helper to create a dummy `TypeId` for tests
    fn dummy_ty() -> TypeChildId {
        TypeChildId::Nat(NatId::new(0))
    }

    /// Helper to create dummy nat nodes hashmap with a "0" leaf at NatId(0)
    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut nats = HashMap::new();
        nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        nats
    }

    /// Helper to convert a Vec of `EClasses` to a `HashMap` with sequential `EClassIds`
    fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
        classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (EClassId::new(i), c))
            .collect()
    }

    /// Helper to build a simple graph with one class containing one node
    fn single_node_graph(label: &str) -> EGraph<String> {
        EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf(label.to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        )
    }

    #[test]
    fn enumerate_single_leaf() {
        let graph = single_node_graph("a");
        let trees = graph.enumerate_trees(0, true);

        assert_eq!(trees.len(), 1);
        // with_type=true wraps in typeOf(expr, type)
        assert_eq!(trees[0].label(), "typeOf");
        assert_eq!(trees[0].children().len(), 2);
        assert_eq!(trees[0].children()[0].label(), "a"); // expr
        assert_eq!(trees[0].children()[1].label(), "0"); // type
    }

    #[test]
    fn enumerate_with_or_choice() {
        // Graph with one class containing two node choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let trees = graph.enumerate_trees(0, true);
        assert_eq!(trees.len(), 2);

        // with_type=true wraps in typeOf(expr, type)
        // Extract expr labels from inside the typeOf wrapper
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

        let trees = graph.enumerate_trees(0, true);
        assert_eq!(trees.len(), 1);
        // with_type=true wraps in typeOf(expr, type)
        // typeOf(a(typeOf(b, type), typeOf(c, type)), type)
        assert_eq!(trees[0].label(), "typeOf");
        let expr = &trees[0].children()[0];
        assert_eq!(expr.label(), "a");
        assert_eq!(expr.children().len(), 2); // b and c (each wrapped in typeOf)
        assert_eq!(expr.children()[0].children()[0].label(), "b");
        assert_eq!(expr.children()[1].children()[0].label(), "c");
        assert_eq!(trees[0].children()[1].label(), "0"); // a's type
    }

    #[test]
    fn enumerate_with_cycle_no_revisits() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("a".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 0 revisits, we can only take the leaf option
        let trees = graph.enumerate_trees(0, true);
        assert_eq!(trees.len(), 1);
        // with_type=true wraps in typeOf(expr, type)
        assert_eq!(trees[0].label(), "typeOf");
        assert_eq!(trees[0].children()[0].label(), "leaf");
    }

    #[test]
    fn enumerate_with_cycle_one_revisit() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("rec".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 1 revisit, we can go one level deep
        let trees = graph.enumerate_trees(1, true);

        // Should have: "leaf", "rec(leaf)", "rec(rec(leaf))"
        // Actually: at depth 0 we have 2 choices, at depth 1 we have 2 choices...
        // Let's verify we get more trees than with 0 revisits
        assert!(trees.len() > 1);

        // with_type=true wraps in typeOf(expr, type)
        // Check that we have the recursive structure
        let has_recursive = trees
            .iter()
            .any(|t| t.label() == "typeOf" && t.children()[0].label() == "rec");
        assert!(has_recursive);
    }

    #[test]
    fn min_distance_exact_match() {
        // Graph contains the exact reference tree
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

        // Reference: with_type=true wraps in typeOf(expr, type)
        // typeOf(a(typeOf(b, type), typeOf(c, type)), type)
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
                leaf("0".to_owned()), // a's type
            ],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
    }

    #[test]
    fn min_distance_chooses_best() {
        // Graph with OR choice: "a" or "x"
        // Reference is "a", so should choose "a" with distance 0
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

        // Reference: with_type=true wraps in typeOf(expr, type)
        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
        // Result is wrapped in typeOf
        assert_eq!(result.0.label(), "typeOf");
        assert_eq!(result.0.children()[0].label(), "a");
    }

    #[test]
    fn min_distance_with_structure_choice() {
        // Graph offers two structures:
        // Option 1: a(b)
        // Option 2: a(b, c)
        // Reference: a(b)
        // Should choose option 1 with distance 0
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("a".to_owned(), vec![eid(1)]),         // a(b)
                        ENode::new("a".to_owned(), vec![eid(1), eid(2)]), // a(b, c)
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

        // Reference: with_type=true wraps in typeOf(expr, type)
        // typeOf(a(typeOf(b, type)), type)
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
                leaf("0".to_owned()), // a's type
            ],
        );
        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();

        assert_eq!(result.1, 0);
        // Result is typeOf(a(...), type), so outer node has 2 children
        assert_eq!(result.0.children().len(), 2);
        // Inner 'a' has 1 child (b wrapped in typeOf)
        assert_eq!(result.0.children()[0].children().len(), 1);
    }

    #[test]
    fn tree_from_choices_single_leaf() {
        let graph = single_node_graph("a");
        let choices = vec![0];

        let tree = graph.tree_from_choices(EClassId::new(0), &choices, true);

        // with_type=true wraps in typeOf(expr, type)
        assert_eq!(tree.label(), "typeOf");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "a"); // expr
        assert_eq!(tree.children()[1].label(), "0"); // type
    }

    #[test]
    fn tree_from_choices_or_choice_first() {
        // Graph with two OR choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices, true);

        // with_type=true wraps in typeOf(expr, type)
        assert_eq!(tree.label(), "typeOf");
        assert_eq!(tree.children()[0].label(), "a");
    }

    #[test]
    fn tree_from_choices_or_choice_second() {
        // Graph with two OR choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![1];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices, true);

        // with_type=true wraps in typeOf(expr, type)
        assert_eq!(tree.label(), "typeOf");
        assert_eq!(tree.children()[0].label(), "b");
    }

    #[test]
    fn tree_from_choices_with_and_children() {
        // Graph: root -> node with two child classes
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

        let choices = vec![0, 0, 0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices, true);

        // with_type=true wraps in typeOf(expr, type)
        // typeOf(a(typeOf(b, type), typeOf(c, type)), type)
        assert_eq!(tree.label(), "typeOf");
        let expr = &tree.children()[0];
        assert_eq!(expr.label(), "a");
        assert_eq!(expr.children().len(), 2); // b, c (each wrapped in typeOf)
        assert_eq!(expr.children()[0].children()[0].label(), "b");
        assert_eq!(expr.children()[1].children()[0].label(), "c");
        assert_eq!(tree.children()[1].label(), "0"); // a's type
    }

    #[test]
    fn tree_from_choices_nested_or_choices() {
        // Graph with nested OR choices:
        // Class 0: root with two node options
        //   Node "x" -> points to Class 1
        //   Node "y" -> points to Class 1
        // Class 1: two leaf options "a" or "b"
        let graph = EGraph::new(
            cfv(vec![
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
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // with_type=true wraps each node in typeOf(expr, type)
        // Test typeOf(x(typeOf(a, type)), type)
        let choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1, true);
        assert_eq!(tree1.label(), "typeOf");
        assert_eq!(tree1.children()[0].label(), "x");
        // x's child is typeOf(a, type)
        assert_eq!(tree1.children()[0].children()[0].children()[0].label(), "a");

        // Test typeOf(x(typeOf(b, type)), type)
        let choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2, true);
        assert_eq!(tree2.label(), "typeOf");
        assert_eq!(tree2.children()[0].label(), "x");
        assert_eq!(tree2.children()[0].children()[0].children()[0].label(), "b");

        // Test typeOf(y(typeOf(a, type)), type)
        let choices3 = vec![1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3, true);
        assert_eq!(tree3.label(), "typeOf");
        assert_eq!(tree3.children()[0].label(), "y");
        assert_eq!(tree3.children()[0].children()[0].children()[0].label(), "a");

        // Test typeOf(y(typeOf(b, type)), type)
        let choices4 = vec![1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4, true);
        assert_eq!(tree4.label(), "typeOf");
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
            cfv(vec![
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
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // with_type=true wraps each node in typeOf(expr, type)
        // Test typeOf(p(typeOf(a, type), typeOf(x, type)), type)
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1, true);
        assert_eq!(tree1.label(), "typeOf");
        assert_eq!(tree1.children()[0].label(), "p");
        assert_eq!(tree1.children()[0].children()[0].children()[0].label(), "a");
        assert_eq!(tree1.children()[0].children()[1].children()[0].label(), "x");

        // Test typeOf(p(typeOf(a, type), typeOf(y, type)), type)
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2, true);
        assert_eq!(tree2.label(), "typeOf");
        assert_eq!(tree2.children()[0].label(), "p");
        assert_eq!(tree2.children()[0].children()[0].children()[0].label(), "a");
        assert_eq!(tree2.children()[0].children()[1].children()[0].label(), "y");

        // Test typeOf(p(typeOf(b, type), typeOf(x, type)), type)
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3, true);
        assert_eq!(tree3.label(), "typeOf");
        assert_eq!(tree3.children()[0].label(), "p");
        assert_eq!(tree3.children()[0].children()[0].children()[0].label(), "b");
        assert_eq!(tree3.children()[0].children()[1].children()[0].label(), "x");

        // Test typeOf(p(typeOf(b, type), typeOf(y, type)), type)
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4, true);
        assert_eq!(tree4.label(), "typeOf");
        assert_eq!(tree4.children()[0].label(), "p");
        assert_eq!(tree4.children()[0].children()[0].children()[0].label(), "b");
        assert_eq!(tree4.children()[0].children()[1].children()[0].label(), "y");
    }

    #[test]
    fn tree_from_choices_deep_nesting() {
        // Deep tree with choices at multiple levels
        // Class 0: root -> "a" with child [Class 1]
        // Class 1: "b1" or "b2", both with child [Class 2]
        // Class 2: "c1" or "c2"
        let graph = EGraph::new(
            cfv(vec![
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
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // with_type=true wraps each node in typeOf(expr, type)
        // Test typeOf(a(typeOf(b1(typeOf(c1, type)), type)), type)
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1, true);
        assert_eq!(tree1.label(), "typeOf");
        let a = &tree1.children()[0];
        assert_eq!(a.label(), "a");
        // a's child is typeOf(b1(...), type)
        let b1_wrapped = &a.children()[0];
        assert_eq!(b1_wrapped.label(), "typeOf");
        let b1 = &b1_wrapped.children()[0];
        assert_eq!(b1.label(), "b1");
        // b1's child is typeOf(c1, type)
        let c1_wrapped = &b1.children()[0];
        assert_eq!(c1_wrapped.children()[0].label(), "c1");

        // Test typeOf(a(typeOf(b1(typeOf(c2, type)), type)), type)
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2, true);
        assert_eq!(tree2.label(), "typeOf");
        assert_eq!(tree2.children()[0].label(), "a");
        assert_eq!(
            tree2.children()[0].children()[0].children()[0].label(),
            "b1"
        );
        assert_eq!(
            tree2.children()[0].children()[0].children()[0].children()[0].children()[0].label(),
            "c2"
        );

        // Test typeOf(a(typeOf(b2(typeOf(c1, type)), type)), type)
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3, true);
        assert_eq!(tree3.label(), "typeOf");
        assert_eq!(tree3.children()[0].label(), "a");
        assert_eq!(
            tree3.children()[0].children()[0].children()[0].label(),
            "b2"
        );
        assert_eq!(
            tree3.children()[0].children()[0].children()[0].children()[0].children()[0].label(),
            "c1"
        );

        // Test typeOf(a(typeOf(b2(typeOf(c2, type)), type)), type)
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4, true);
        assert_eq!(tree4.label(), "typeOf");
        assert_eq!(tree4.children()[0].label(), "a");
        assert_eq!(
            tree4.children()[0].children()[0].children()[0].label(),
            "b2"
        );
        assert_eq!(
            tree4.children()[0].children()[0].children()[0].children()[0].children()[0].label(),
            "c2"
        );
    }

    #[test]
    fn tree_from_choices_three_and_children() {
        // Test with three AND children
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2), eid(3)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0, 0, 0, 0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices, true);

        // with_type=true wraps each node in typeOf(expr, type)
        assert_eq!(tree.label(), "typeOf");
        let f = &tree.children()[0];
        assert_eq!(f.label(), "f");
        assert_eq!(f.children().len(), 3); // a, b, c (each wrapped in typeOf)
        assert_eq!(f.children()[0].children()[0].label(), "a");
        assert_eq!(f.children()[1].children()[0].label(), "b");
        assert_eq!(f.children()[2].children()[0].label(), "c");
        assert_eq!(tree.children()[1].label(), "0"); // f's type
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
            cfv(vec![
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
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let enumerated = graph.enumerate_trees(0, true);

        // Should produce: x(a), x(b), y
        assert_eq!(enumerated.len(), 3);

        // Reconstruct using tree_from_choices
        let choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1, true);
        assert!(trees_equal(&tree1, &enumerated[0]));

        let choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2, true);
        assert!(trees_equal(&tree2, &enumerated[1]));

        let choices3 = vec![1];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3, true);
        assert!(trees_equal(&tree3, &enumerated[2]));
    }

    #[test]
    fn deserialize_json_file() {
        let graph = EGraph::<String>::parse_from_file(Path::new(
            "data/rise/egraph_jsons/ser_egraph_vectorization_SRL_step_2_iteration_0_root_150.json",
        ));

        // Verify root is correct
        assert_eq!(graph.root().to_index(), 150);

        // Verify we can access the root class
        let root_class = graph.class(graph.root());
        assert!(!root_class.nodes().is_empty());

        // Verify some classes exist
        assert!(!graph.classes.is_empty());

        // Verify nat nodes exist
        assert!(!graph.nat_nodes.is_empty());
    }

    #[test]
    fn min_distance_extract_fast_exact_match() {
        // Graph contains the exact reference tree
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

        // Reference: with_type=true wraps in typeOf(expr, type)
        // typeOf(a(typeOf(b, type), typeOf(c, type)), type)
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
        // Graph with OR choice: "a" or "x"
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

        // Reference: with_type=true wraps in typeOf(expr, type)
        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let result = find_min_zs(&graph, &reference, &UnitCost, 0, true)
            .0
            .unwrap();
        assert_eq!(result.1, 0);
        // Result is wrapped in typeOf
        assert_eq!(result.0.label(), "typeOf");
        assert_eq!(result.0.children()[0].label(), "a");
    }

    #[test]
    fn min_distance_extract_filtered_prunes_bad_trees() {
        // Create a graph where one option is clearly worse
        // Option 1: typeOf(a, type) - matches reference exactly
        // Option 2: typeOf(x, type) - has different label, lower bound >= 1
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

        // Reference: typeOf(a, type) - tree_from_choices wraps in typeOf
        let reference = node(
            "typeOf".to_owned(),
            vec![leaf("a".to_owned()), leaf("0".to_owned())],
        );

        let (result, stats) = find_min_zs(&graph, &reference, &UnitCost, 0, true);

        assert_eq!(result.unwrap().1, 0);
        assert_eq!(stats.trees_enumerated, 2);
        // With parallel execution, pruning is non-deterministic since both trees
        // may be processed before best_distance is updated. We just verify the
        // invariant that pruned + full_comparisons == trees_enumerated.
        assert_eq!(
            stats.size_pruned + stats.euler_pruned + stats.full_comparisons,
            stats.trees_enumerated
        );
    }
}
