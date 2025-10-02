mod ast_size;
mod contains;
mod extract;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::utils::UniqueQueue;

pub(crate) use contains::{SatisfiesContainsAnalysis, SatisfiesOnlyContainsAnalysis};
pub(crate) use extract::{ExtractAnalysis, ExtractContainsAnalysis, ExtractOnlyContainsAnalysis};

pub trait SemiLatticeAnalysis<L, N>: Sized + Debug
where
    L: Language,
    N: Analysis<L>,
{
    type Data: Debug;

    fn make<'a>(
        &mut self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data
    where
        Self::Data: 'a,
        Self: 'a;

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&mut self, egraph: &EGraph<L, N>, data: &mut HashMap<Id, Self::Data>) {
        fn resolve_pending_analysis<L: Language, A: Analysis<L>, B: SemiLatticeAnalysis<L, A>>(
            egraph: &EGraph<L, A>,
            analysis: &mut B,
            data: &mut HashMap<Id, B::Data>,
            analysis_pending: &mut UniqueQueue<Id>,
        ) {
            // Take the next node from the worklist
            while let Some(node_id) = analysis_pending.pop() {
                // Legal thanks to https://docs.rs/egg/latest/egg/struct.EGraph.html#method.nodes
                let node = egraph.nodes()[usize::from(node_id)].clone();
                let u_node = node.map_children(|id| egraph.find(id));

                // If we have data for all the children
                if u_node.all(|child_id| data.contains_key(&child_id)) {
                    let canonical_eclass_id = egraph.find(node_id);
                    let eclass = &egraph[canonical_eclass_id];
                    // We make the analysis for this node
                    let node_data = analysis.make(egraph, &u_node, data);
                    if let Some(existing) = data.get_mut(&canonical_eclass_id) {
                        // If we already have data about this node, we need to update it
                        let DidMerge(may_not_be_existing, _) = analysis.merge(existing, node_data);
                        // If this changed anything, we need to re-evaluate the parents,
                        // until we have reached the fixpoint
                        if may_not_be_existing {
                            analysis_pending.extend(eclass.parents());
                        }
                    } else {
                        // If we have no data about the node, we add the new data and
                        // then add the parents to worklist
                        data.insert(canonical_eclass_id, node_data);
                        analysis_pending.extend(eclass.parents());
                    }
                } else {
                    // If we don't have data about this, put it back on the queue and try later
                    analysis_pending.insert(node_id);
                }
            }
        }
        assert!(egraph.clean);
        let mut analysis_pending = UniqueQueue::default();

        for (index, enode) in egraph.nodes().iter().enumerate() {
            if enode.all(|c| data.contains_key(&egraph.find(c))) {
                // Adding all the enodes to the worklist to start the analysis
                // If we have perfect knowledge of their children.
                // This is always true for leave nodes
                // Legal thanks to https://docs.rs/egg/latest/egg/struct.EGraph.html#method.nodes
                analysis_pending.insert(Id::from(index));
            }
        }

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}
