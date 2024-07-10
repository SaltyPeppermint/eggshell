use std::fmt::Debug;

use egg::{AstDepth, AstSize, CostFunction, Language};

#[derive(Clone, Copy, Debug)]
pub struct AstSize2;

impl<L: Language> CostFunction<L> for AstSize2 {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &L, costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let mut inner = AstSize;
        inner.cost(enode, costs)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AstDepth2;

impl<L: Language> CostFunction<L> for AstDepth2 {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &L, costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let mut inner = AstDepth;
        inner.cost(enode, costs)
    }
}
