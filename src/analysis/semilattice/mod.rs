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
        fn resolve_pending_analysis<'a, L, N, B>(
            egraph: &'a EGraph<L, N>,
            analysis: &mut B,
            data: &mut HashMap<Id, B::Data>,
            analysis_pending: &mut UniqueQueue<Id>,
        ) where
            L: Language,
            N: Analysis<L>,
            B: SemiLatticeAnalysis<L, N>,
        {
            while let Some(current_id) = analysis_pending.pop() {
                let node = egraph.nodes()[usize::from(current_id)].clone();
                let u_node = node.map_children(|id| egraph.find(id)); // find_mut?

                if u_node.all(|id| data.contains_key(&id)) {
                    let canonical_id = egraph.find(current_id); // find_mut?
                    let eclass = &egraph[canonical_id];
                    let node_data = analysis.make(egraph, &u_node, data);
                    let new_data = match data.remove(&canonical_id) {
                        None => {
                            analysis_pending.extend(eclass.parents());
                            node_data
                        }
                        Some(mut existing) => {
                            let DidMerge(may_not_be_existing, _) =
                                analysis.merge(&mut existing, node_data);
                            if may_not_be_existing {
                                analysis_pending.extend(eclass.parents());
                            }
                            existing
                        }
                    };
                    data.insert(canonical_id, new_data);
                } else {
                    analysis_pending.insert(current_id);
                }
            }
        }

        assert!(egraph.clean);
        let mut analysis_pending = UniqueQueue::default();

        for (index, enode) in egraph.nodes().iter().enumerate() {
            if enode.all(|c| data.contains_key(&egraph.find(c))) {
                analysis_pending.insert(Id::from(index));
            }
        }

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}
