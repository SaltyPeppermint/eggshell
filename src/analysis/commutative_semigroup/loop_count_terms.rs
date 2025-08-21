use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;

use dashmap::DashMap;
use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::CommutativeSemigroupAnalysis;
use super::Counter;

// DONT USE THIS MAY BE BUGGY

#[derive(Debug, Copy, Clone)]
pub struct ExprCount {
    limit: usize,
}

impl ExprCount {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<C, L, N> CommutativeSemigroupAnalysis<L, N, C> for ExprCount
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, HashMap<BTreeSet<Id>, C>>;

    /// DONT USE, MAY BE BUGGY
    fn make(
        &self,
        _egraph: &EGraph<L, N>,
        eclass_id: Id,
        enode: &L,
        analysis_of: &Arc<DashMap<Id, Self::Data>>,
    ) -> Self::Data {
        fn rec<CC: Counter>(
            remaining: &[HashMap<usize, HashMap<BTreeSet<Id>, CC>>],
            size: usize,
            count: CC,
            visited: BTreeSet<Id>,
            counts: &mut HashMap<usize, HashMap<BTreeSet<Id>, CC>>,
            limit: usize,
        ) {
            // If we have reached the term size limit, stop the recursion
            if size > limit {
                return;
            }

            // If we have not reached the bottom of the recursion, call rec for all the children
            // with increased size (since a child adds a node) and multiplied count to account for all
            // the possible new combinations.
            // If we have reached the end of the recursion, we add (or init) the count to the entry
            // for the corresponding size
            if let Some((head, tail)) = remaining.split_first() {
                for (s, m) in head {
                    for (v, c) in m {
                        let new_v = visited.union(v).cloned().collect();
                        rec(
                            tail,
                            size + s, //size.checked_add(*s).expect("Add failed in rec"),
                            count.clone() * c, //count.checked_mul(*c).expect("Mul failed in rec"),
                            new_v,
                            counts,
                            limit,
                        );
                    }
                }
            } else {
                if let Some(m_v) = counts.get_mut(&size) {
                    if let Some(c) = m_v.get_mut(&visited) {
                        *c += &count;
                    } else {
                        m_v.insert(visited, count);
                    }
                } else {
                    counts.insert(size, [(visited, count)].into());
                }
            }
        }

        // We start with the initial analysis of all the children since we need to know those
        // so we can simply add their sizes / multiply their counts for this node.
        // Thankfully this is cached via `analysis_of`

        let children_counts = enode
            .children()
            .iter()
            .map(|c_id| analysis_of.get(c_id).unwrap().clone())
            .collect::<Vec<_>>();
        let mut counts = HashMap::new();
        // DONT USE, MAY BE BUGGY
        rec(
            &children_counts,
            1,
            C::one(),
            [(eclass_id)].into(),
            &mut counts,
            self.limit,
        );
        counts
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if b.is_empty() {
            return DidMerge(false, false);
        }

        for (size, b_visited_map) in b {
            if let Some(a_visited_map) = a.get_mut(&size) {
                for (visited, b_count) in b_visited_map {
                    if let Some(a_count) = a_visited_map.get_mut(&visited) {
                        *a_count += b_count;
                    } else {
                        a_visited_map.insert(visited, b_count);
                    }
                }
            } else {
                a.insert(size, b_visited_map);
            }
        }
        DidMerge(true, false)
    }

    // fn new_data() -> Self::Data {
    //     Self::Data::new()
    // }

    // fn data_empty(data: &Self::Data) -> bool {
    //     data.is_empty()
    // }
}
