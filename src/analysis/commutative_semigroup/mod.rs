mod expr_count;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;
use rayon::prelude::*;

use crate::utils::UniqueQueue;

pub use expr_count::Counter;
pub use expr_count::ExprCount;

pub trait CommutativeSemigroupAnalysis<C, L, N>: Sized + Debug + Sync + Send
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
{
    type Data: PartialEq + Debug + Sync + Send;

    fn make<'a>(
        &self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &(impl Fn(egg::Id) -> &'a Self::Data + Sync),
    ) -> Self::Data
    where
        Self::Data: 'a,
        C: 'a;

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&self, egraph: &EGraph<L, N>, data: &mut HashMap<Id, Self::Data>) {
        fn resolve_pending_analysis<CC, L, N, B>(
            egraph: &EGraph<L, N>,
            analysis: &B,
            data: &mut HashMap<Id, B::Data>,
            analysis_pending: &mut UniqueQueue<Id>,
        ) where
            L: Language + Debug + Sync + Send,
            L::Discriminant: Debug + Sync,
            N: Analysis<L> + Debug + Sync,
            N::Data: Debug + Sync,
            B: CommutativeSemigroupAnalysis<CC, L, N> + Sync + Send,
            B::Data: PartialEq + Debug,
        {
            while let Some(id) = analysis_pending.pop() {
                let canonical_id = egraph.find(id);
                debug_assert_eq!(canonical_id, id);
                let eclass = &egraph[canonical_id];

                // Check if we can calculate the analysis for any enode
                let available_data = eclass.nodes.par_iter().filter_map(|n| {
                    let u_node = n.clone().map_children(|child_id| egraph.find(child_id));
                    // If all the childs eclass_children have data, we can calculate it!
                    if u_node.all(|child_id| data.contains_key(&child_id)) {
                        Some(analysis.make(egraph, &u_node, &|child_id| &data[&child_id]))
                    } else {
                        None
                    }
                });

                // If we have some info, we add that info to our storage.
                // Otherwise we have absolutely no info about the nodes so we can only put them back onto the queue.
                // and hope for a better time later.
                if let Some(computed_data) = available_data.reduce_with(|mut a, b| {
                    analysis.merge(&mut a, b);
                    a
                }) {
                    // If we have gained new information, put the parents onto the queue.
                    // They need to be re-evaluated.
                    // Only once we have reached a fixpoint we can stop updating the parents.
                    if !(data.get(&eclass.id) == Some(&computed_data)) {
                        analysis_pending.extend(eclass.parents().map(|p| egraph.find(p)));
                        data.insert(eclass.id, computed_data);
                    }
                } else {
                    assert!(!eclass.nodes.is_empty());
                    analysis_pending.insert(canonical_id);
                }
            }
        }

        assert!(egraph.clean);

        // We start at the leaves, since they have no children and can be directly evaluated.
        let mut analysis_pending = egraph
            .classes()
            .filter(|eclass| eclass.nodes.iter().any(|enode| enode.is_leaf()))
            // No egraph.find since we are taking the id directly from the eclass
            .map(|eclass| eclass.id)
            .collect();

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}
