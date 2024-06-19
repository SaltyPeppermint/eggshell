use std::collections::VecDeque;
use std::fmt::Display;
use std::hash::Hash;

use egg::{Analysis, EGraph, Language, RecExpr};
use hashbrown::HashMap as HashBrownMap;
use hashbrown::HashSet as HashBrownSet;
use indexmap::IndexMap;
// use indexmap::IndexMap;

use crate::cost_fn::{Cost, CostFn, INFINITY};
use crate::eqsat::ClassId;
use crate::extraction::{ExtractResult, Extractor};

pub struct BottomUp<'a, C, L, N>
where
    C: CostFn,
    L: Language,
    N: Analysis<L>,
{
    cost_fn: C,
    choices: HashBrownMap<ClassId, (&'a L, Cost)>,
    egraph: &'a EGraph<L, N>,
}

impl<'a, C, L, N> BottomUp<'a, C, L, N>
where
    C: CostFn,
    L: Language,
    N: Analysis<L>,
{
    /// Find the cheapest (lowest cost) represented [`egg::RecExpr`] in the
    /// given eclass.
    /// Technically this would also work with non-roots but it isnt tested.
    fn find_best(&self, root: ClassId) -> (RecExpr<L>, Cost) {
        let (root, cost) = self.choices[&self.egraph.find(root)];
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (expr, cost)
    }

    /// Find the cheapest e-node in the given e-class.
    fn find_best_node(&self, eclass: ClassId) -> &L {
        self.choices[&self.egraph.find(eclass)].0
    }
}

impl<'a, C, L, N> Extractor<'a, C, L, N> for BottomUp<'a, C, L, N>
where
    C: CostFn,
    L: Language + Display + Sync,
    N: Analysis<L> + Sync,
{
    fn new(cost_fn: C, egraph: &'a EGraph<L, N>) -> Self {
        BottomUp {
            cost_fn,
            choices: HashBrownMap::new(),
            egraph,
        }
    }

    /// Extract the best terms from the roots
    fn extract(mut self, roots: &[ClassId]) -> Vec<ExtractResult<L>>
    where
        N: Analysis<L>,
    {
        let mut analysis_pending = UniqueQueue::default();

        let mut parents = IndexMap::with_capacity(self.egraph.classes().len());

        for class in self.egraph.classes() {
            parents.insert(class.id, Vec::new());
        }

        // for class in egraph.classes().values() {
        //     for node in &class.nodes {
        //         for c in &egraph[node].children {
        //             // compute parents of this enode
        //             parents[n2c(c)].push(node.clone());
        //         }

        //         // start the analysis from leaves
        //         if egraph[node].is_leaf() {
        //             analysis_pending.insert(node.clone());
        //         }
        //     }
        // }

        // Add all the leave enodes with their class id and position in the eclass
        // to the queue that needs the
        for class in self.egraph.classes() {
            for node in &class.nodes {
                let canonical_id = self.egraph.find(class.id);

                for child_class_id in node.children() {
                    // compute parents of this enode
                    let parent_nodes = parents
                        .get_mut(child_class_id)
                        .expect("no entry found for key");
                    parent_nodes.push((canonical_id, node));
                }

                if node.is_leaf() {
                    analysis_pending.insert((canonical_id, node));
                }
            }
        }

        while let Some((class_id, node)) = analysis_pending.pop() {
            // If the class has previously been rated, it has a cost, otherwise its
            // cost is infinity.
            let prev_cost = match self.choices.get(&class_id) {
                Some((_, cost)) => cost,
                None => &INFINITY,
            };
            // Recalculate the cost of the class if the node of the current loop were chosen.
            let cost = self
                .cost_fn
                .node_sum_cost(self.egraph, class_id, node, &self.choices);
            // If the currently evaluated node is a cheaper choice, the class cost is
            // updated with the new lower cost.
            if cost < *prev_cost {
                // Update the choices one would make if prompted with the nodes
                // eclass were chosen and its cost
                self.choices.insert(class_id, (node, cost));
                // All the parent enodes of this eclass need to be reevaluated since
                // to propagate the new cheaper cost upwards
                analysis_pending.extend(parents[&class_id].clone());
            }
        }
        roots
            .iter()
            .map(|root| self.find_best(*root).into())
            .collect()
    }
}

/// A data structure to maintain a queue of unique elements.
///
/// Notably, insert/pop operations have O(1) expected amortized runtime complexity.
/// Thanks @Bastacyclop for the implementation!
struct UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    set: HashBrownSet<T>,
    queue: VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: HashBrownSet::default(),
            queue: VecDeque::new(),
        }
    }
}

impl<T> UniqueQueue<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter {
            self.insert(t);
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}
