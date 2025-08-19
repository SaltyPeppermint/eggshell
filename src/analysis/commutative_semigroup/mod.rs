mod expr_count;
mod loop_cause;
mod loop_free_count;

use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::sync::Arc;
use std::thread;

use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;
use log::debug;
use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::utils::UniqueQueue;

pub use expr_count::ExprCount;
pub use loop_cause::LoopCause;
pub use loop_free_count::LoopFreeCount;

pub trait CommutativeSemigroupAnalysis<L, N, C = ()>: Sized + Debug + Sync + Send
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    type Data: PartialEq + Debug + Sync + Send;

    fn make(
        &self,
        egraph: &EGraph<L, N>,
        eclass_id: Id,
        enode: &L,
        analysis_of: &Arc<DashMap<Id, Self::Data>>,
    ) -> Self::Data;

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&self, egraph: &EGraph<L, N>) -> HashMap<Id, Self::Data> {
        fn resolve_pending_analysis<'aa, L, N, B, CC>(
            egraph: &EGraph<L, N>,
            analysis: &B,
            data: &Arc<DashMap<Id, B::Data>>,
            queue: &SegQueue<Id>,
        ) where
            L: Language + Sync + Send,
            L::Discriminant: Sync,
            N: Analysis<L> + Debug + Sync,
            N::Data: Debug + Sync,
            B: CommutativeSemigroupAnalysis<L, N, CC> + Sync + Send,
            B::Data: PartialEq + Debug,
        {
            // Potentially, this might lead to a situation where only one thread is working on the queue.
            // This has not been observed in practice, but it is a potential bottleneck.
            while let Some(id) = queue.pop() {
                let canonical_id = egraph.find(id);
                debug_assert_eq!(canonical_id, id);
                let eclass = &egraph[canonical_id];

                // Check if we can calculate the analysis for any enode
                let available_data = eclass.nodes.iter().filter_map(|n| {
                    let u_node = n.clone().map_children(|child_id| egraph.find(child_id));
                    // If all the childs eclass_children have data, we can calculate it!
                    u_node
                        .all(|child_id| data.contains_key(&child_id))
                        .then(|| analysis.make(egraph, canonical_id, &u_node, data))
                });

                // If we have some info, we add that info to our storage.
                // Otherwise we have absolutely no info about the nodes so we can only put them back onto the queue.
                // and hope for a better time later.
                if let Some(computed_data) = available_data.reduce(|mut a, b| {
                    analysis.merge(&mut a, b);
                    a
                }) {
                    // If we have gained new information, put the parents onto the queue.
                    // They need to be re-evaluated.
                    // Only once we have reached a fixpoint we can stop updating the parents.
                    if !(data
                        .get(&eclass.id)
                        .is_some_and(|v| v.value() == &computed_data))
                    {
                        for i in eclass.parents().map(|p| egraph.find(p)) {
                            queue.push(i);
                        }
                        data.insert(eclass.id, computed_data);
                    }
                } else {
                    assert!(!eclass.nodes.is_empty());
                    queue.push(canonical_id);
                }
            }
        }

        assert!(egraph.clean);

        // We start at the leaves, since they have no children and can be directly evaluated.

        let queue = Arc::new(SegQueue::new());

        for id in egraph
            .classes()
            .filter(|eclass| eclass.nodes.iter().any(|enode| enode.is_leaf()))
            .map(|eclass| eclass.id)
        {
            queue.push(id);
        }

        let par_data = Arc::new(DashMap::new());
        let pd = thread::scope(move |scope| {
            for i in 0..thread::available_parallelism().unwrap().get() {
                let thread_data = par_data.clone();
                let thread_queue = queue.clone();

                scope.spawn(move || {
                    debug!("Thread #{i} started!");
                    resolve_pending_analysis(egraph, self, &thread_data, &thread_queue);
                    debug!("Thread #{i} finished!");
                });
            }
            par_data
        });

        let data = Arc::into_inner(pd)
            .unwrap()
            .into_par_iter()
            .collect::<HashMap<_, _>>();

        debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
        data
    }
}

pub trait Counter:
    Debug
    + Clone
    + Send
    + Sync
    + Debug
    + NumRef
    + NumAssignRef
    + for<'x> Sum<&'x Self>
    + Sum<Self>
    + for<'x> Product<&'x Self>
    + Product<Self>
    + SampleUniform
    + PartialOrd
    + Default
{
}
impl<
    T: Debug
        + Clone
        + Send
        + Sync
        + Debug
        + NumRef
        + NumAssignRef
        + for<'x> Sum<&'x Self>
        + Sum<Self>
        + for<'x> Product<&'x Self>
        + Product<Self>
        + SampleUniform
        + PartialOrd
        + Default,
> Counter for T
{
}
