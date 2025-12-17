use egg::{Analysis, DidMerge, EGraph, Id, Language, RecExpr};
use hashbrown::HashSet;

use super::nat::try_simplify;
use super::{DBIndex, Kindable, Rise};

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

#[derive(Default, Debug, Clone)]
pub struct AnalysisData {
    pub free: HashSet<DBIndex>,
    pub beta_extract: RecExpr<Rise>,
    pub canon_nat_expr: RecExpr<Rise>,
}

impl Analysis<Rise> for RiseAnalysis {
    type Data = AnalysisData;

    fn merge(&mut self, to: &mut AnalysisData, from: AnalysisData) -> DidMerge {
        let before_len = to.free.len();
        to.free.extend(from.free);
        let free_changed = before_len != to.free.len();

        let beta_changed = if !from.beta_extract.is_empty()
            && (to.beta_extract.is_empty() || to.beta_extract.len() > from.beta_extract.len())
        {
            to.beta_extract = from.beta_extract;
            true
        } else {
            false
        };

        let nat_changed = if !from.canon_nat_expr.is_empty()
            && (to.canon_nat_expr.is_empty() || to.canon_nat_expr.len() > from.canon_nat_expr.len())
        {
            to.canon_nat_expr = from.canon_nat_expr;
            true
        } else {
            false
        };

        // TODO: more precise second bool
        DidMerge(free_changed || beta_changed || nat_changed, true)
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

        let empty = enode.any(|id| egraph[id].data.beta_extract.is_empty());

        // let beta_extract = if empty {
        //     RecExpr::default()
        // } else {
        //     enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        // };
        let beta_extract = if empty {
            RecExpr::default()
        } else {
            enode.join_recexprs(|id| egraph[id].data.beta_extract.as_ref())
        };

        let canon_nat_expr = if beta_extract.is_empty() {
            RecExpr::default()
        } else {
            try_simplify(&beta_extract).unwrap_or_default()
        };

        AnalysisData {
            free,
            beta_extract,
            canon_nat_expr,
        }
    }

    fn modify(egraph: &mut EGraph<Rise, RiseAnalysis>, id: Id) {
        if !egraph[id].data.canon_nat_expr.is_empty()
            && egraph[id].data.canon_nat_expr != egraph[id].data.beta_extract
        {
            // Remove all other nodes, only the canonical one may remain.
            egraph[id].nodes.clear();
            // Add the canonical expr
            let canon_nat = &egraph[id].data.canon_nat_expr;
            let added = egraph.add_expr(&canon_nat.clone());
            egraph.union(id, added);

            // #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}
