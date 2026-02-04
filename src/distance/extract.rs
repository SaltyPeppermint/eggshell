//! Tree extraction for E-Graphs.
//!
//! This module provides iterators for enumerating trees from an e-graph.

use hashbrown::HashMap;

use super::graph::EGraph;
use super::ids::{EClassId, ExprChildId};
use super::nodes::Label;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::graph::EClass;
    use crate::distance::ids::{ExprChildId, NatId, TypeChildId};
    use crate::distance::nodes::{ENode, NatNode};

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
