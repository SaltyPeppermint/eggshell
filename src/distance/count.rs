//! Term counting analysis for e-graphs.
//!
//! Counts the number of terms up to a given size that can be extracted from each e-class.

use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use dashmap::DashMap;
use hashbrown::{HashMap, HashSet};
use log::debug;
use num_traits::{NumAssignRef, NumRef};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::graph::EGraph;
use super::ids::{DataChildId, DataId, EClassId, ExprChildId, FunId, NatId, TypeChildId};
use super::nodes::Label;
use crate::utils::UniqueQueue;

/// Counter trait for counting terms.
pub trait Counter: Clone + Send + Sync + NumRef + NumAssignRef + Default + std::fmt::Debug {}

impl<T: Clone + Send + Sync + NumRef + NumAssignRef + Default + std::fmt::Debug> Counter for T {}

/// Configuration for the term counting analysis.
#[derive(Debug, Clone)]
pub struct TermCount {
    limit: usize,
    with_types: bool,
}

impl TermCount {
    /// Create a new term counting configuration.
    ///
    /// # Arguments
    /// * `limit` - Maximum term size to count
    /// * `with_types` - If true, include type annotations in size calculations
    #[must_use]
    pub fn new(limit: usize, with_types: bool) -> Self {
        Self { limit, with_types }
    }

    /// Run the term counting analysis on an e-graph.
    ///
    /// Returns a map from e-class ID to a map of (size -> count).
    ///
    /// # Panics
    /// Panics if threads fail to join (should not happen in practice).
    #[must_use]
    pub fn analyze<L: Label, C: Counter>(
        &self,
        egraph: &EGraph<L>,
    ) -> HashMap<EClassId, HashMap<usize, C>> {
        // Build parent map and type size cache
        let parents = build_parent_map(egraph);
        let type_cache = Arc::new(RwLock::new(TypeSizeCache::default()));

        // Find leaf classes (classes with at least one leaf node)
        let leaves: UniqueQueue<EClassId> = egraph
            .class_ids()
            .filter(|&id| {
                egraph
                    .class(id)
                    .nodes()
                    .iter()
                    .any(|n| n.children().is_empty())
            })
            .collect();

        let analysis_pending = Arc::new(Mutex::new(leaves));
        let data: Arc<DashMap<EClassId, HashMap<usize, C>>> = Arc::new(DashMap::new());

        // Run parallel analysis
        let result_data = thread::scope(|scope| {
            for i in 0..thread::available_parallelism().map_or(1, |p| p.get()) {
                let thread_data = data.clone();
                let thread_pending = analysis_pending.clone();
                let thread_type_cache = type_cache.clone();
                let thread_parents = &parents;

                scope.spawn(move || {
                    debug!("Thread #{i} started!");
                    self.resolve_pending_analysis(
                        egraph,
                        &thread_data,
                        &thread_pending,
                        &thread_type_cache,
                        thread_parents,
                    );
                    debug!("Thread #{i} finished!");
                });
            }
            data
        });

        Arc::into_inner(result_data)
            .unwrap()
            .into_par_iter()
            .collect()
    }

    /// Process pending e-classes from the work queue.
    fn resolve_pending_analysis<L: Label, C: Counter>(
        &self,
        egraph: &EGraph<L>,
        data: &Arc<DashMap<EClassId, HashMap<usize, C>>>,
        analysis_pending: &Arc<Mutex<UniqueQueue<EClassId>>>,
        type_cache: &Arc<RwLock<TypeSizeCache>>,
        parents: &HashMap<EClassId, HashSet<EClassId>>,
    ) {
        // Mirrors the structure of CommutativeSemigroupAnalysis::one_shot_analysis
        while let Some(id) = { analysis_pending.lock().unwrap().pop() } {
            let canonical_id = egraph.canonicalize(id);
            let eclass = egraph.class(canonical_id);

            // Get the type overhead for this e-class
            let type_overhead = if self.with_types {
                let ty_size = TypeSizeCache::get_type_size(type_cache, egraph, eclass.ty());
                1 + ty_size
            } else {
                0
            };

            // Check if we can calculate the analysis for any enode
            let available_data = eclass.nodes().iter().filter_map(|node| {
                // If all the eclass children have data, we can calculate it
                let all_ready = node.children().iter().all(|child_id| match child_id {
                    ExprChildId::Nat(_) | ExprChildId::Data(_) => true,
                    ExprChildId::EClass(eclass_id) => {
                        data.contains_key(&egraph.canonicalize(*eclass_id))
                    }
                });
                all_ready.then(|| {
                    self.make_node_data(egraph, node.children(), data, type_cache, type_overhead)
                })
            });

            // If we have some info, we add that info to our storage.
            // Otherwise put back onto the queue.
            if let Some(computed_data) = available_data.reduce(|mut a, b| {
                Self::merge(&mut a, b);
                a
            }) {
                // If we have gained new information, put the parents onto the queue.
                // Only once we have reached a fixpoint we can stop updating the parents.
                if !(data.get(&canonical_id).is_some_and(|v| *v == computed_data)) {
                    if let Some(parent_set) = parents.get(&canonical_id) {
                        analysis_pending
                            .lock()
                            .unwrap()
                            .extend(parent_set.iter().copied());
                    }
                    data.insert(canonical_id, computed_data);
                }
            } else {
                assert!(!eclass.nodes().is_empty());
                analysis_pending.lock().unwrap().insert(canonical_id);
            }
        }
    }

    /// Merge two term count data maps.
    fn merge<C: Counter>(a: &mut HashMap<usize, C>, b: HashMap<usize, C>) {
        for (size, count) in b {
            a.entry(size).and_modify(|c| *c += &count).or_insert(count);
        }
    }

    /// Compute term counts for a single e-node.
    fn make_node_data<L: Label, C: Counter>(
        &self,
        egraph: &EGraph<L>,
        children: &[ExprChildId],
        data: &Arc<DashMap<EClassId, HashMap<usize, C>>>,
        type_cache: &Arc<RwLock<TypeSizeCache>>,
        type_overhead: usize,
    ) -> HashMap<usize, C> {
        // Base size: 1 for the node itself + type overhead
        let base_size = 1 + type_overhead;

        if children.is_empty() {
            // Leaf node
            if base_size <= self.limit {
                return HashMap::from([(base_size, C::one())]);
            }
            return HashMap::new();
        }

        // For nodes with children, combine their counts
        let mut tmp = Vec::new();

        children.iter().fold(
            HashMap::from([(base_size, C::one())]),
            |mut acc, child_id| {
                let child_data = Self::get_child_data::<L, C>(egraph, *child_id, data, type_cache);

                tmp.extend(acc.drain());

                for (acc_size, acc_count) in &tmp {
                    for (child_size, child_count) in &child_data {
                        let combined_size = acc_size + child_size;
                        if combined_size > self.limit {
                            continue;
                        }
                        let combined_count = acc_count.to_owned() * child_count;
                        acc.entry(combined_size)
                            .and_modify(|c| *c += &combined_count)
                            .or_insert(combined_count);
                    }
                }

                tmp.clear();
                acc
            },
        )
    }

    /// Get the count data for a child, handling Nat/Data/EClass variants.
    fn get_child_data<L: Label, C: Counter>(
        egraph: &EGraph<L>,
        child_id: ExprChildId,
        data: &Arc<DashMap<EClassId, HashMap<usize, C>>>,
        type_cache: &Arc<RwLock<TypeSizeCache>>,
    ) -> HashMap<usize, C> {
        match child_id {
            ExprChildId::Nat(nat_id) => {
                // Nat nodes have a fixed size (no choices)
                let size = TypeSizeCache::get_nat_size(type_cache, egraph, nat_id);
                let mut result = HashMap::new();
                result.insert(size, C::one());
                result
            }
            ExprChildId::Data(data_id) => {
                // Data type nodes have a fixed size (no choices)
                let size = TypeSizeCache::get_data_size(type_cache, egraph, data_id);
                let mut result = HashMap::new();
                result.insert(size, C::one());
                result
            }
            ExprChildId::EClass(eclass_id) => {
                // E-class children use the precomputed data
                let canonical_id = egraph.canonicalize(eclass_id);
                data.get(&canonical_id)
                    .map(|r| r.clone())
                    .unwrap_or_default()
            }
        }
    }
}

/// Build a map from child e-class to parent e-classes.
fn build_parent_map<L: Label>(egraph: &EGraph<L>) -> HashMap<EClassId, HashSet<EClassId>> {
    let mut parents: HashMap<EClassId, HashSet<EClassId>> = HashMap::new();

    for class_id in egraph.class_ids() {
        let canonical_id = egraph.canonicalize(class_id);
        for node in egraph.class(canonical_id).nodes() {
            for child_id in node.children() {
                if let ExprChildId::EClass(child_eclass_id) = child_id {
                    let canonical_child = egraph.canonicalize(*child_eclass_id);
                    parents
                        .entry(canonical_child)
                        .or_default()
                        .insert(canonical_id);
                }
            }
        }
    }

    parents
}

/// Cache for type node sizes to avoid repeated computation.
#[derive(Debug, Default)]
struct TypeSizeCache {
    nats: HashMap<NatId, usize>,
    data: HashMap<DataId, usize>,
    funs: HashMap<FunId, usize>,
}

impl TypeSizeCache {
    /// Get the size of a type (`TypeChildId`), dispatching to the appropriate cache.
    fn get_type_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: TypeChildId) -> usize {
        match id {
            TypeChildId::Nat(nat_id) => Self::get_nat_size(cache, egraph, nat_id),
            TypeChildId::Type(fun_id) => Self::get_fun_size(cache, egraph, fun_id),
            TypeChildId::Data(data_id) => Self::get_data_size(cache, egraph, data_id),
        }
    }

    /// Get the size of a nat node, using cache with read-preferring access.
    fn get_nat_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: NatId) -> usize {
        // Try read lock first (fast path for cache hits)
        if let Some(&size) = cache.read().unwrap().nats.get(&id) {
            return size;
        }

        // Cache miss: compute and insert with write lock
        let size = Self::compute_nat_size(cache, egraph, id);
        cache.write().unwrap().nats.insert(id, size);
        size
    }

    /// Get the size of a data type node, using cache with read-preferring access.
    fn get_data_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: DataId) -> usize {
        // Try read lock first (fast path for cache hits)
        if let Some(&size) = cache.read().unwrap().data.get(&id) {
            return size;
        }

        // Cache miss: compute and insert with write lock
        let size = Self::compute_data_size(cache, egraph, id);
        cache.write().unwrap().data.insert(id, size);
        size
    }

    /// Get the size of a fun type node, using cache with read-preferring access.
    fn get_fun_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: FunId) -> usize {
        // Try read lock first (fast path for cache hits)
        if let Some(&size) = cache.read().unwrap().funs.get(&id) {
            return size;
        }

        // Cache miss: compute and insert with write lock
        let size = Self::compute_fun_size(cache, egraph, id);
        cache.write().unwrap().funs.insert(id, size);
        size
    }

    fn compute_nat_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: NatId) -> usize {
        let node = egraph.nat(id);
        let children_size: usize = node
            .children()
            .iter()
            .map(|&child_id| Self::get_nat_size(cache, egraph, child_id))
            .sum();
        1 + children_size
    }

    fn compute_data_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: DataId) -> usize {
        let node = egraph.data_ty(id);
        let children_size: usize = node
            .children()
            .iter()
            .map(|&child_id| match child_id {
                DataChildId::Nat(nat_id) => Self::get_nat_size(cache, egraph, nat_id),
                DataChildId::DataType(data_id) => Self::get_data_size(cache, egraph, data_id),
            })
            .sum();
        1 + children_size
    }

    fn compute_fun_size<L: Label>(cache: &RwLock<Self>, egraph: &EGraph<L>, id: FunId) -> usize {
        let node = egraph.fun_ty(id);
        let children_size: usize = node
            .children()
            .iter()
            .map(|&child_id| Self::get_type_size(cache, egraph, child_id))
            .sum();
        1 + children_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::graph::EClass;
    use crate::distance::nodes::{ENode, NatNode};
    use num::BigUint;

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
    fn single_leaf_no_types() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        let root_data = &data[&EClassId::new(0)];
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&1], BigUint::from(1u32));
    }

    #[test]
    fn single_leaf_with_types() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, true);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        let root_data = &data[&EClassId::new(0)];
        // Size = 1 (node) + 1 (typeOf) + 1 (type "0") = 3
        assert_eq!(root_data.len(), 1);
        assert_eq!(root_data[&3], BigUint::from(1u32));
    }

    #[test]
    fn two_choices_no_types() {
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        let root_data = &data[&EClassId::new(0)];
        // Two terms of size 1
        assert_eq!(root_data[&1], BigUint::from(2u32));
    }

    #[test]
    fn parent_child_no_types() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: has leaf "a"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        // Class 1: one term of size 1
        assert_eq!(data[&EClassId::new(1)][&1], BigUint::from(1u32));

        // Class 0: one term of size 2 (f + a)
        assert_eq!(data[&EClassId::new(0)][&2], BigUint::from(1u32));
    }

    #[test]
    fn parent_with_multiple_child_choices() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: has two leaves "a" and "b"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        // Class 1: two terms of size 1
        assert_eq!(data[&EClassId::new(1)][&1], BigUint::from(2u32));

        // Class 0: two terms of size 2 (f(a), f(b))
        assert_eq!(data[&EClassId::new(0)][&2], BigUint::from(2u32));
    }

    #[test]
    fn two_children() {
        // Class 0: has node "f" pointing to classes 1 and 2
        // Class 1: leaf "a"
        // Class 2: leaf "b"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        // Class 0: one term of size 3 (f + a + b)
        assert_eq!(data[&EClassId::new(0)][&3], BigUint::from(1u32));
    }

    #[test]
    fn combinatorial_explosion() {
        // Class 0: has node "f" pointing to classes 1 and 2
        // Class 1: two leaves "a1", "a2"
        // Class 2: three leaves "b1", "b2", "b3"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a1".to_owned()), ENode::leaf("a2".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![
                        ENode::leaf("b1".to_owned()),
                        ENode::leaf("b2".to_owned()),
                        ENode::leaf("b3".to_owned()),
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

        let counter = TermCount::new(10, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        // Class 0: 2 * 3 = 6 terms of size 3
        assert_eq!(data[&EClassId::new(0)][&3], BigUint::from(6u32));
    }

    #[test]
    fn size_limit_filters() {
        // Class 0: has node "f" pointing to class 1
        // Class 1: leaf "a"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("f".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Limit = 1, so f(a) with size 2 should be filtered out
        let counter = TermCount::new(1, false);
        let data: HashMap<EClassId, HashMap<usize, BigUint>> = counter.analyze(&graph);

        // Class 1 should have data (size 1)
        assert!(data.contains_key(&EClassId::new(1)));
        assert_eq!(data[&EClassId::new(1)][&1], BigUint::from(1u32));

        // Class 0 should be empty (size 2 exceeds limit)
        assert!(data.get(&EClassId::new(0)).is_none_or(|d| d.is_empty()));
    }
}
