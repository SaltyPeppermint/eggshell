use std::cmp::{Ordering, PartialOrd};

use egg::{Analysis, DidMerge, EGraph, Id, Language, RecExpr};
use hashbrown::HashSet;

use super::Rise;
use super::db::Index;
use super::kind::Kindable;

#[derive(Default, Debug)]
pub struct RiseAnalysis {
    nat_eq_cache: EGraph<Rise, ()>,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Optimal {
    cost: usize,
    node: Rise,
}

impl Ord for Optimal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
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
        let before_len = to.free.len();
        to.free.extend(from.free);
        let free_changed = DidMerge(before_len != to.free.len(), true);

        let optimal_changed = egg::merge_option(&mut to.optimal, from.optimal, egg::merge_min);

        free_changed | optimal_changed //|| beta_changed || nat_changed
    }

    fn make(egraph: &mut EGraph<Rise, RiseAnalysis>, enode: &Rise, _: Id) -> AnalysisData {
        let free = match enode {
            Rise::Var(v) => [*v].into(),
            Rise::Lambda(l, e) => egraph[*e]
                .data
                .free
                .iter()
                .filter(|i| !i.is_zero() && i.kind() == l.kind())
                .map(|i| i.dec(l.kind()))
                .collect(),
            _ => enode
                .children()
                .iter()
                .flat_map(|c| egraph[*c].data.free.iter())
                .copied()
                .collect(),
        };

        let optimal = enode
            .children()
            .iter()
            .map(|id| egraph[*id].data.optimal.as_ref().map(|o| o.cost))
            .sum::<Option<usize>>()
            .map(|c_cost| Optimal {
                cost: c_cost + 1,
                node: enode.to_owned(),
            });

        AnalysisData { free, optimal }
    }

    // fn modify(egraph: &mut EGraph<Rise, FreeBetaNatAnalysis>, id: Id) {
    //     if !egraph[id].data.canon_nat_expr.is_empty()
    //         && egraph[id].data.canon_nat_expr != egraph[id].data.beta_extract
    //     {
    //         // Add the canonical expr
    //         let canon_nat = &egraph[id].data.canon_nat_expr;
    //         let added = egraph.add_expr(&canon_nat.clone());
    //         egraph.union(id, added);

    //         #[cfg(debug_assertions)]
    //         egraph[id].assert_unique_leaves();
    //     }
    // }
}
