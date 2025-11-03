use egg::{CostFunction, Id, Language};
use num::{Integer, One};

use super::{EGraph, Math};

// cost function similar to AstSize except it will
// penalize `(pow _ p)` where p is a fraction
pub struct AltCost<'a> {
    pub egraph: &'a EGraph,
}

impl<'a> AltCost<'a> {
    pub fn new(egraph: &'a EGraph) -> Self {
        Self { egraph }
    }
}

impl CostFunction<Math> for AltCost<'_> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        if let Math::Pow([_, i]) = enode
            && let Some((n, _reason)) = &self.egraph[*i].data
            && !n.denom().is_one()
            && n.denom().is_odd()
        {
            return usize::MAX;
        }

        enode.fold(1, |sum, id| usize::saturating_add(sum, costs(id)))
    }
}
