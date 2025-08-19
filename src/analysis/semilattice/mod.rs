mod ast_size;
mod contains;
mod extract;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::old_parents_iter;
use crate::utils::UniqueQueue;

pub(crate) use contains::SatisfiesContainsAnalysis;
pub use extract::ExtractAnalysis;

pub trait SemiLatticeAnalysis<L: Language, N: Analysis<L>>: Sized + Debug {
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
        fn resolve_pending_analysis<
            'a,
            L: Language,
            N: Analysis<L>,
            B: SemiLatticeAnalysis<L, N>,
        >(
            egraph: &'a EGraph<L, N>,
            analysis: &mut B,
            data: &mut HashMap<Id, B::Data>,
            analysis_pending: &mut UniqueQueue<(&'a L, Id)>,
        ) {
            while let Some((node, current_id)) = analysis_pending.pop() {
                let u_node = node.clone().map_children(|child_id| egraph.find(child_id));

                if u_node.all(|id| data.contains_key(&id)) {
                    // No egraph.find since since analysis_pending only contains canonical ids
                    let eclass = &egraph[current_id];
                    let node_data = analysis.make(egraph, &u_node, data);
                    if let Some(existing) = data.get_mut(&current_id) {
                        let DidMerge(may_not_be_existing, _) = analysis.merge(existing, node_data);
                        if may_not_be_existing {
                            analysis_pending.extend(old_parents_iter(eclass, egraph));
                        }
                    } else {
                        // old_parents_iter returns only canonical ids
                        analysis_pending.extend(old_parents_iter(eclass, egraph));
                        data.insert(current_id, node_data);
                    }
                } else {
                    analysis_pending.insert((node, current_id));
                }
            }
        }

        assert!(egraph.clean);

        let mut analysis_pending = egraph
            .classes()
            .flat_map(|eclass| {
                eclass
                    .nodes
                    .iter()
                    .filter(|enode| enode.all(|c| data.contains_key(&egraph.find(c))))
                    // No egraph.find since we are taking the id directly from the eclass
                    .map(|enode| (enode, eclass.id))
            })
            .collect();

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}
