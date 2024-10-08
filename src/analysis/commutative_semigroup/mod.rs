mod count_terms;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::utils::UniqueQueue;

pub use count_terms::TermsUpToSize;

pub trait CommutativeSemigroupAnalysis<L: Language, N: Analysis<L>>: Sized + Debug {
    type Data: PartialEq + Debug;

    fn make<'a>(
        &mut self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
        Self: 'a;

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&mut self, egraph: &EGraph<L, N>, data: &mut HashMap<Id, Self::Data>) {
        fn resolve_pending_analysis<L, N, B>(
            egraph: &EGraph<L, N>,
            analysis: &mut B,
            data: &mut HashMap<Id, B::Data>,
            analysis_pending: &mut UniqueQueue<Id>,
        ) where
            L: Language,
            N: Analysis<L>,
            B: CommutativeSemigroupAnalysis<L, N>,
            B::Data: PartialEq + Debug,
        {
            while let Some(id) = analysis_pending.pop() {
                let canonical_id = egraph.find(id);
                debug_assert!(canonical_id == id);
                let eclass = &egraph[canonical_id];

                // Check if we can calculate the analysis for any enode
                let available_data = eclass
                    .nodes
                    .iter()
                    .filter_map(|n| {
                        let u_node = n.clone().map_children(|child_id| egraph.find(child_id));
                        // If all the childs eclass_children have data, we can calculate it!
                        if u_node.all(|child_id| data.contains_key(&child_id)) {
                            Some(analysis.make(egraph, &u_node, &|child_id| &data[&child_id]))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                // If we have some info, we add that info to our storage.
                // Otherwise, if we have absolutely now info about the nodes, we can only put them back onto the queue.
                // and hope for a better time later.
                if let Some(computed_data) = available_data.into_iter().reduce(|mut a, b| {
                    analysis.merge(&mut a, b);
                    a
                }) {
                    // If we have gained new information, put the parents onto the queue.
                    // They need to be re-evaluated.
                    // Only once we have reached a fixpoint we can stop updating the parents.
                    if !(data.get(&eclass.id) == Some(&computed_data)) {
                        analysis_pending.extend(eclass.parents().map(|p| egraph.find(p.1)));
                        data.insert(eclass.id, computed_data);
                    }
                } else {
                    // assert!(eclass.nodes.is_empty());
                    // FIXME
                    // OK WHY DO I NEED TO COMMENT OUT THE EMPTY CHECK
                    analysis_pending.insert(canonical_id);
                }
            }
        }

        assert!(egraph.clean);

        let mut analysis_pending = UniqueQueue::<Id>::default();

        // We start at the leaves, since they have no children and can be directly evaluated.
        for eclass in egraph.classes() {
            for enode in &eclass.nodes {
                if enode.is_leaf() {
                    analysis_pending.insert(eclass.id);
                }
            }
        }

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}
