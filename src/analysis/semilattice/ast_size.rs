use std::cmp::Ordering;

use egg::{Analysis, AstSize, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::SemiLatticeAnalysis;

impl<L: Language, N: Analysis<L>> SemiLatticeAnalysis<L, N> for AstSize {
    type Data = usize;

    fn make<'a>(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data
    where
        Self::Data: 'a,
    {
        enode.fold(1usize, |size, id| size + analysis_of[&id])
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        match (*a).cmp(&b) {
            Ordering::Less => DidMerge(false, true),
            Ordering::Equal => DidMerge(false, false),
            Ordering::Greater => {
                *a = b;
                DidMerge(true, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use egg::SymbolLang;
    use hashbrown::HashMap;

    use super::*;

    #[test]
    fn simple_analysis() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let z = egraph.add(SymbolLang::leaf("0"));
        let apz = egraph.add(SymbolLang::new("+", vec![a, z]));

        egraph.union(a, apz);
        egraph.rebuild();

        let mut data = HashMap::default();
        AstSize.one_shot_analysis(&egraph, &mut data);

        assert_eq!(data[&egraph.find(apz)], 1);
    }
}
