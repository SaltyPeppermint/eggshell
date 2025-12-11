use egg::{Analysis, DidMerge, EGraph, Id, Language, RecExpr};
use hashbrown::HashSet;

use crate::rewrite_system::rise::kind::Kindable;
use crate::rewrite_system::rise::nat::try_simplify;

use super::{DBIndex, Rise};

#[derive(Default, Debug)]
pub struct RiseAnalysis {
    nat_eq_cache: EGraph<Rise, ()>,
}

impl RiseAnalysis {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nat_eq_cache: EGraph::default(),
        }
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

#[derive(Default, Debug)]
pub struct AnalysisData {
    pub free: HashSet<DBIndex>,
    pub beta_extract: RecExpr<Rise>,
    // pub simple_nat: Option<RecExpr<Rise>>,
}

impl Analysis<Rise> for RiseAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, to: &mut AnalysisData, from: AnalysisData) -> DidMerge {
        let before_len = to.free.len();
        to.free.extend(from.free);
        let free_merge = DidMerge(before_len != to.free.len(), true);

        let beta_merge = if !from.beta_extract.is_empty()
            && (to.beta_extract.is_empty() || to.beta_extract.len() > from.beta_extract.len())
        {
            to.beta_extract = from.beta_extract;
            DidMerge(true, true)
        } else {
            DidMerge(false, true) // TODO: more precise second bool
        };

        // let nat_merge =
        //     egg::merge_option(&mut to.simple_nat, from.simple_nat, |to_nat, from_nat| {
        //         if to_nat.len() > from_nat.len() {
        //             *to_nat = from_nat;
        //             DidMerge(true, true)
        //         } else {
        //             DidMerge(false, true)
        //         }
        //     });

        free_merge | beta_merge // | nat_merge
    }

    fn make(egraph: &mut EGraph<Rise, RiseAnalysis>, enode: &Rise) -> AnalysisData {
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

        let empty = enode.any(|id| egraph[id].data.beta_extract.is_empty());

        let beta_extract = if empty {
            RecExpr::default()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };
        // let (beta_extract, simple_nat) = if empty {
        //     (RecExpr::default(), None)
        // } else {
        //     let expr = enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref());
        //     let simple_nat = try_simplify(&expr).ok();
        //     (expr, simple_nat)
        // };

        AnalysisData {
            free,
            beta_extract,
            // simple_nat,
        }
    }

    // fn modify(egraph: &mut EGraph<Rise, RiseAnalysis>, id: Id) {
    //     if let Some(expr) = egraph[id].data.simple_nat.clone() {
    //         let added = egraph.add_expr(&expr);
    //         egraph.union(id, added);

    //         // to not prune, comment this out
    //         // egraph[id].nodes.retain(egg::Language::is_leaf);

    //         // #[cfg(debug_assertions)]
    //         egraph[id].assert_unique_leaves();
    //     }
    // }
}
