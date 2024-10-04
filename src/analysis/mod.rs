mod ast_size;
mod contains;
mod extract;

use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::utils::{ExprHashCons, UniqueQueue};

pub(crate) use contains::ExtractContainsAnalysis;
pub(crate) use contains::SatisfiesContainsAnalysis;
pub(crate) use extract::ExtractAnalysis;

pub trait SemiLatticeAnalysis<L: Language, N: Analysis<L>>: Sized {
    type Data: Debug;

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
        assert!(egraph.clean);

        let mut analysis_pending = UniqueQueue::<(L, Id)>::default();
        // works with queue but IndexSet is stack
        // IndexSet::<(L, Id)>::default();

        for eclass in egraph.classes() {
            for enode in &eclass.nodes {
                if enode.all(|c| data.contains_key(&egraph.find(c))) {
                    analysis_pending.insert((enode.clone(), eclass.id));
                }
            }
        }

        resolve_pending_analysis(egraph, self, data, &mut analysis_pending);

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
    }
}

fn resolve_pending_analysis<L: Language, N: Analysis<L>, B: SemiLatticeAnalysis<L, N>>(
    egraph: &EGraph<L, N>,
    analysis: &mut B,
    data: &mut HashMap<Id, B::Data>,
    analysis_pending: &mut UniqueQueue<(L, Id)>,
) {
    while let Some((node, current_id)) = analysis_pending.pop() {
        let u_node = node.clone().map_children(|child_id| egraph.find(child_id)); // find_mut?

        if u_node.all(|id| data.contains_key(&id)) {
            let canonical_id = egraph.find(current_id); // find_mut?
            let eclass = &egraph[canonical_id];
            let node_data = analysis.make(egraph, &u_node, &|id| &data[&id]);
            let new_data = match data.remove(&canonical_id) {
                None => {
                    analysis_pending.extend(
                        eclass
                            .parents()
                            .map(|(n, parent_id)| (n.clone(), parent_id)),
                    );
                    node_data
                }
                Some(mut existing) => {
                    let DidMerge(may_not_be_existing, _) = analysis.merge(&mut existing, node_data);
                    if may_not_be_existing {
                        analysis_pending.extend(
                            eclass
                                .parents()
                                .map(|(n, parent_id)| (n.clone(), parent_id)),
                        );
                    }
                    existing
                }
            };
            data.insert(canonical_id, new_data);
        } else {
            analysis_pending.insert((node, current_id));
        }
    }
}
