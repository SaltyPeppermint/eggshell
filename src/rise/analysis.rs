use std::cmp::{Ordering, PartialOrd};

use egg::{Analysis, DidMerge, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

use crate::rise::db::Index;
use crate::rise::kind::Kindable;
use crate::rise::{Rise, canon_nat};

#[derive(Default, Debug)]
pub struct RiseAnalysis {
    nat_eq_cache: EGraph<Rise, ()>,
    min_upstream_size: Option<HashMap<Id, usize>>,
}

impl RiseAnalysis {
    #[must_use]
    pub fn new() -> Self {
        RiseAnalysis::default()
    }

    pub fn check_cache_equiv(&mut self, lhs: &RecExpr<Rise>, rhs: &RecExpr<Rise>) -> bool {
        if let Some(lhs_id) = self.nat_eq_cache.lookup_expr(lhs)
            && let Some(rhs_id) = self.nat_eq_cache.lookup_expr(rhs)
            && lhs_id == rhs_id
        {
            return true;
        }
        false
    }

    pub fn add_pair_to_cache(&mut self, lhs: &RecExpr<Rise>, rhs: &RecExpr<Rise>) {
        let lhs_id = self.nat_eq_cache.add_expr(lhs);
        let rhs_id = self.nat_eq_cache.add_expr(rhs);
        self.nat_eq_cache.union(lhs_id, rhs_id);
    }

    pub fn set_min_upstream_size(&mut self, min_upstream_size: Option<HashMap<Id, usize>>) {
        self.min_upstream_size = min_upstream_size;
    }

    #[must_use]
    pub fn min_upstream_size(&self) -> Option<&HashMap<Id, usize>> {
        self.min_upstream_size.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Optimal {
    size: usize,
    node: Rise,
}

impl Optimal {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn node(&self) -> &Rise {
        &self.node
    }
}

impl Ord for Optimal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.size.cmp(&other.size)
    }
}

impl PartialOrd for Optimal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisData {
    pub free: HashSet<Index>,
    pub optimal: Option<Optimal>,
}

impl AnalysisData {
    pub fn small_repr(&self, egraph: &EGraph<Rise, RiseAnalysis>) -> Option<RecExpr<Rise>> {
        self.optimal
            .as_ref()?
            .node
            .try_build_recexpr(|c_id| {
                egraph[c_id]
                    .data
                    .optimal
                    .as_ref()
                    .map(|o| o.node.clone())
                    .ok_or(())
            })
            .ok()
    }
}

impl Analysis<Rise> for RiseAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, to: &mut AnalysisData, from: AnalysisData) -> DidMerge {
        let is_superset = to.free.is_superset(&from.free);
        to.free.extend(from.free);
        let free_changed = DidMerge(!is_superset, true);

        let optimal_changed = egg::merge_option(&mut to.optimal, from.optimal, egg::merge_min);

        free_changed | optimal_changed //|| beta_changed || nat_changed
    }

    fn make(egraph: &mut EGraph<Rise, RiseAnalysis>, enode: &Rise, _: Id) -> AnalysisData {
        let free = match enode {
            Rise::Var(v) => [*v].into(),
            Rise::Lambda(l, [e, ty]) => egraph[*e]
                .data
                .free
                .iter()
                .filter_map(|i| {
                    if i.kind() == l.kind() {
                        if i.is_zero() { None } else { Some(i.dec()) }
                    } else {
                        Some(*i)
                    }
                })
                .chain(egraph[*ty].data.free.iter().filter_map(|i| {
                    if i.kind() == l.kind() {
                        if i.is_zero() { None } else { Some(i.dec()) }
                    } else {
                        Some(*i)
                    }
                }))
                .collect(),
            _ => enode
                .children()
                .iter()
                .flat_map(|c_id| egraph[*c_id].data.free.iter())
                .copied()
                .collect(),
        };

        let optimal = enode
            .children()
            .iter()
            .map(|id| egraph[*id].data.optimal.as_ref().map(|o| o.size))
            .sum::<Option<usize>>()
            .map(|children_size| Optimal {
                size: children_size + 1,
                node: enode.to_owned(),
            });

        AnalysisData { free, optimal }
    }

    fn modify(egraph: &mut EGraph<Rise, RiseAnalysis>, id: Id) {
        if egraph[id].nodes.iter().any(|n| n.is_nat()) {
            // Add the canonical expr
            let Some(expr) = egraph[id].data.small_repr(egraph) else {
                return;
            };
            let canon_nat_expr = canon_nat(&expr);
            let added = egraph.add_expr(&canon_nat_expr);
            egraph.union(id, added);

            // #[cfg(debug_assertions)]
            // egraph[id].assert_unique_leaves();
        }
    }
}
