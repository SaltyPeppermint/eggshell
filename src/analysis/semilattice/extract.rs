use std::fmt::Debug;

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language};

use super::SemiLatticeAnalysis;
use crate::utils::ExprHashCons;

#[derive(Debug)]
pub struct ExtractAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L> + Debug,
    CF::Cost: Debug,
{
    pub(crate) exprs: &'a mut ExprHashCons<L>,
    pub(crate) cost_fn: &'a mut CF,
}

impl<'a, L, CF> ExtractAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L> + Debug,
    CF::Cost: Debug,
{
    pub(crate) fn new(exprs: &'a mut ExprHashCons<L>, cost_fn: &'a mut CF) -> Self {
        Self { exprs, cost_fn }
    }
}

impl<'a, L, N, CF> SemiLatticeAnalysis<L, N> for ExtractAnalysis<'a, L, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'a,
{
    type Data = (CF::Cost, Id);

    fn make<'b>(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'b Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'b,
        Self: 'b,
    {
        let expr_node = enode.clone().map_children(|c| (*analysis_of)(c).1);
        let expr = self.exprs.add(expr_node);
        (
            self.cost_fn.cost(enode, |c| (*analysis_of)(c).0.clone()),
            expr,
        )
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if a.0 < b.0 {
            DidMerge(false, true)
        } else if a.0 == b.0 {
            DidMerge(false, false)
        } else {
            *a = b;
            DidMerge(true, false)
        }
    }
}
