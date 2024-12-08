use std::fmt::Debug;
use std::ops::AddAssign;
use std::ops::Mul;

use egg::{Analysis, DidMerge, EGraph, Language};
use hashbrown::HashMap;
use num::BigUint;
use rayon::prelude::*;

use super::CommutativeSemigroupAnalysis;

pub trait Counter:
    Debug
    + Clone
    + PartialEq
    + From<u32>
    + for<'x> Mul<&'x Self, Output = Self>
    + AddAssign
    + Send
    + Sync
{
}

impl Counter for f64 {}
impl Counter for BigUint {}

#[derive(Debug)]
pub struct ExprCount {
    limit: usize,
}

impl ExprCount {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<C, L, N> CommutativeSemigroupAnalysis<C, L, N> for ExprCount
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
    C: Counter,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, C>;

    fn make<'a>(
        &self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &(impl Fn(egg::Id) -> &'a Self::Data + Sync),
    ) -> Self::Data
    where
        Self::Data: 'a,
        C: 'a,
        Self: 'a,
    {
        fn rec<CC: Counter>(
            remaining: &[&HashMap<usize, CC>],
            size: usize,
            count: CC,
            counts: &mut HashMap<usize, CC>,
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
            if let Some((head, tail)) = remaining.split_first() {
                for (s, c) in *head {
                    rec(
                        tail,
                        size + s, //size.checked_add(*s).expect("Add failed in rec"),
                        count.clone() * c, //count.checked_mul(*c).expect("Mul failed in rec"),
                        counts,
                        limit,
                    );
                }
            } else {
                // counts
                //     .entry(size)
                //     .and_modify(|c| {
                //         *c += count;
                //     })
                //     .or_insert(count);
                match counts.get_mut(&size) {
                    Some(c) => *c += count, // .checked_add(count).expect("Add failed in else"),
                    None => {
                        counts.insert(size, count);
                    }
                }
            }
        }
        // We start with the initial analysis of all the children since we need to know those
        // so we can simply add their sizes / multiply their counts for this node.
        // Thankfully this is cached via `analysis_of`
        let children_counts = enode
            .children()
            .par_iter()
            .map(|c_id| analysis_of(*c_id))
            .collect::<Vec<_>>();
        let mut counts = HashMap::new();

        rec(&children_counts, 1, 1u32.into(), &mut counts, self.limit);
        counts
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if b.is_empty() {
            return DidMerge(false, false);
        }

        for (size, count) in b {
            // a.entry(size)
            //     .and_modify(|c| {
            //         *c += count;
            //     })
            //     .or_insert(count);
            match a.get_mut(&size) {
                Some(c) => *c += count, //  .checked_add(count).expect("Add failed in merge"),
                None => {
                    a.insert(size, count);
                }
            }
        }
        DidMerge(true, false)
    }

    // fn new_data() -> Self::Data {
    //     Self::Data::new()
    // }

    // fn data_empty(data: &Self::Data) -> bool {
    //     data.is_empty()
    // }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};
    use num::BigUint;

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::trs::{Halide, TermRewriteSystem};

    use super::*;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let mut data = HashMap::<_, HashMap<_, BigUint>>::new();
        ExprCount::new(10).one_shot_analysis(&egraph, &mut data);

        let root_data = &data[&egraph.find(apb)];

        assert_eq!(root_data[&5], 1u32.into());
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

        let mut data = HashMap::<_, HashMap<_, BigUint>>::new();
        ExprCount::new(10).one_shot_analysis(&egraph, &mut data);

        let root_data = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16u32.into());
    }

    #[test]
    fn halide_count_size() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(5).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let egraph = eqsat.egraph();
        let root = eqsat.roots()[0];

        let mut data = HashMap::<_, HashMap<_, BigUint>>::new();
        ExprCount::new(16).one_shot_analysis(egraph, &mut data);

        let root_data = &data[&egraph.find(root)];

        assert_eq!(root_data[&16], 40512u32.into());
    }
}
