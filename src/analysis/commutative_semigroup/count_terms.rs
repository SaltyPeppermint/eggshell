use egg::{Analysis, DidMerge, EGraph, Language};
use hashbrown::HashMap;

use super::CommutativeSemigroupAnalysis;

#[derive(Debug)]
pub struct TermsUpToSize {
    limit: usize,
}

impl TermsUpToSize {
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<L, N> CommutativeSemigroupAnalysis<L, N> for TermsUpToSize
where
    L: Language,
    N: Analysis<L>,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, usize>;

    fn make<'a>(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(egg::Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
        Self: 'a,
    {
        fn rec(
            remaining: &[&HashMap<usize, usize>],
            size: usize,
            count: usize,
            counts: &mut HashMap<usize, usize>,
            limit: usize,
        ) {
            // If we have reached the term size limit, stop the recursion
            if size > limit {
                return;
            }

            // If we have not reached the bottom of the recursion, call rec for all the children
            // with increased size (since a child adds a node) and multiplied count to account for all
            // the possible new combinations.
            // If we have reached the bottom, we add (or init) the count to entry corresponding to the current
            // the recursion depth (which is the size)
            if let Some((head, rest)) = remaining.split_first() {
                for (s, c) in *head {
                    rec(rest, size + s, count * c, counts, limit);
                }
            } else {
                counts
                    .entry(size)
                    .and_modify(|c| {
                        *c += count;
                    })
                    .or_insert(count);
            }
        }
        // We start with the initial analysis of all the children since we need to know those
        // so we can simply add their sizes / multiply their counts for this node.
        // Thankfully this is cached via `analysis_of`
        let children_counts = enode
            .children()
            .iter()
            .map(|c_id| analysis_of(*c_id))
            .collect::<Vec<_>>();
        let mut counts = HashMap::new();
        rec(&children_counts, 1, 1, &mut counts, self.limit);
        counts
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if b.is_empty() {
            return DidMerge(false, false);
        }

        for (size, count) in b {
            a.entry(size)
                .and_modify(|c| {
                    *c += count;
                })
                .or_insert(count);
        }
        DidMerge(true, false)
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};

    use super::*;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        // dbg!(&egraph);

        let mut data = HashMap::new();
        TermsUpToSize::new(10).one_shot_analysis(&egraph, &mut data);

        let root_data = &data[&egraph.find(apb)];

        assert_eq!(root_data[&5], 1);
    }

    #[test]
    fn slightly_complicated_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        egraph.union(b, apb);
        egraph.rebuild();
        // dbg!(&egraph);

        let mut data = HashMap::new();
        TermsUpToSize::new(10).one_shot_analysis(&egraph, &mut data);

        let root_data = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16);
    }
}
