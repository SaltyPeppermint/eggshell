use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};

use super::SemiLatticeAnalysis;

#[derive(Debug)]
pub struct SatisfiesContainsAnalysis;

impl<L: Language, N: Analysis<L>> SemiLatticeAnalysis<L, N> for SatisfiesContainsAnalysis {
    type Data = bool;

    fn make<'a>(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
    {
        enode.any(|c| *analysis_of(c))
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let r = *a || b;
        let dm = DidMerge(r != *a, r != b);
        *a = r;
        dm
    }
}
