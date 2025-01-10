mod expr_count;

use std::fmt::Debug;
use std::iter::{Product, Sum};

use num_traits::{NumAssignRef, NumRef};
use rand::distributions::uniform::SampleUniform;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;
use log::debug;

use crate::utils::UniqueQueue;

pub use expr_count::ExprCount;

pub trait CommutativeSemigroupAnalysis<C, L, N>: Sized + Debug + Sync + Send
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    type Data: PartialEq + Debug + Sync + Send;

    fn make<'a>(
        &self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &Arc<RwLock<HashMap<Id, Self::Data>>>,
    ) -> Self::Data
    where
        Self::Data: 'a,
        C: 'a;

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge;

    fn one_shot_analysis(&self, egraph: &EGraph<L, N>) -> HashMap<Id, Self::Data> {
        fn resolve_pending_analysis<CC, L, N, B>(
            egraph: &EGraph<L, N>,
            analysis: &B,
            data: &Arc<RwLock<HashMap<Id, B::Data>>>,
            analysis_pending: &Arc<Mutex<UniqueQueue<Id>>>,
        ) where
            L: Language + Sync + Send,
            L::Discriminant: Sync,
            N: Analysis<L> + Debug + Sync,
            N::Data: Debug + Sync,
            B: CommutativeSemigroupAnalysis<CC, L, N> + Sync + Send,
            B::Data: PartialEq + Debug,
        {
            while let Some(id) = {
                // Potentially, this might lead to a situation where only one thread is working on the queue.
                // This has not been observed in practice, but it is a potential bottleneck.
                let id = { analysis_pending.lock().unwrap().pop() };
                id
            } {
                let canonical_id = egraph.find(id);
                debug_assert_eq!(canonical_id, id);
                let eclass = &egraph[canonical_id];

                // Check if we can calculate the analysis for any enode
                let available_data = eclass.nodes.iter().filter_map(|n| {
                    let u_node = n.clone().map_children(|child_id| egraph.find(child_id));
                    // If all the childs eclass_children have data, we can calculate it!
                    if u_node.all(|child_id| {
                        let a = { data.read().unwrap().contains_key(&child_id) };
                        a
                    }) {
                        Some(analysis.make(egraph, &u_node, data))
                    } else {
                        None
                    }
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
                    if !(data.read().unwrap().get(&eclass.id) == Some(&computed_data)) {
                        analysis_pending
                            .lock()
                            .unwrap()
                            .extend(eclass.parents().map(|p| egraph.find(p)));
                        data.write().unwrap().insert(eclass.id, computed_data);
                    }
                } else {
                    assert!(!eclass.nodes.is_empty());
                    analysis_pending.lock().unwrap().insert(canonical_id);
                }
            }
        }

        assert!(egraph.clean);

        // We start at the leaves, since they have no children and can be directly evaluated.
        let analysis_pending = egraph
            .classes()
            .filter(|eclass| eclass.nodes.iter().any(|enode| enode.is_leaf()))
            // No egraph.find since we are taking the id directly from the eclass
            .map(|eclass| eclass.id)
            .collect();

        let par_analysis_pending = Arc::new(Mutex::new(analysis_pending));

        let par_data = Arc::new(RwLock::new(HashMap::new()));
        thread::scope(|s| {
            for i in 0..thread::available_parallelism().unwrap().get() {
                let thread_data = par_data.clone();
                let thread_analysis_pending = par_analysis_pending.clone();
                s.spawn(move || {
                    debug!("Thread #{i} started!");
                    resolve_pending_analysis(egraph, self, &thread_data, &thread_analysis_pending);
                    debug!("Thread #{i} finished!");
                });
            }
        });

        let data = Arc::into_inner(par_data).unwrap().into_inner().unwrap();

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
