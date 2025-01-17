use std::fmt::Debug;
use std::sync::Arc;
use std::sync::RwLock;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::CommutativeSemigroupAnalysis;
use super::Counter;

#[derive(Debug, Copy, Clone)]
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
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, C>;

    fn make<'a>(
        &self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &Arc<RwLock<HashMap<Id, Self::Data>>>,
    ) -> Self::Data
    where
        Self::Data: 'a,
        C: 'a,
    {
        // fn rec<
        //     CC: Debug
        //         + Clone
        //         + PartialEq
        //         + From<u32>
        //         + for<'x> Mul<&'x CC, Output = CC>
        //         + for<'x> AddAssign<&'x CC>
        //         + Send
        //         + Sync,
        // >(
        //     remaining: &[&HashMap<usize, CC>],
        //     size: usize,
        //     count: CC,
        //     counts: &mut HashMap<usize, CC>,
        //     limit: usize,
        // ) {
        //     // If we have reached the term size limit, stop the recursion
        //     if size > limit {
        //         return;
        //     }

        //     // If we have not reached the bottom of the recursion, call rec for all the children
        //     // with increased size (since a child adds a node) and multiplied count to account for all
        //     // the possible new combinations.
        //     // If we have reached the end of the recursion, we add (or init) the count to the entry
        //     // for the corresponding size
        //     if let Some((head, tail)) = remaining.split_first() {
        //         for (s, c) in *head {
        //             rec(
        //                 tail,
        //                 size + s, //size.checked_add(*s).expect("Add failed in rec"),
        //                 count.clone() * c, //count.checked_mul(*c).expect("Mul failed in rec"),
        //                 counts,
        //                 limit,
        //             );
        //         }
        //     } else {
        //         counts
        //             .entry(size)
        //             .and_modify(|c| {
        //                 *c += &count;
        //             })
        //             .or_insert(count);
        //         // match counts.get_mut(&size) {
        //         //     Some(c) => *c += count, // .checked_add(count).expect("Add failed in else"),
        //         //     None => {
        //         //         counts.insert(size, count);
        //         //     }
        //         // }
        //     }
        // }

        // We start with the initial analysis of all the children since we need to know those
        // so we can simply add their sizes / multiply their counts for this node.
        // Thankfully this is cached via `analysis_of`

        // let children_counts = enode
        //     .children()
        //     .iter()
        //     .map(|c_id| &analysis_of[c_id])
        //     .collect::<Vec<_>>();
        // let mut counts = HashMap::new();
        // rec(&children_counts, 1, 1u32.into(), &mut counts, self.limit);
        // counts

        let mut children_data = Vec::new();
        {
            let a_o = analysis_of.read().unwrap();
            for child_id in enode.children() {
                children_data.push(a_o[child_id].clone());
            }
        }

        let mut tmp = Vec::new();

        children_data
            .into_iter()
            .fold(HashMap::from([(1, C::one())]), |mut acc, child_data| {
                tmp.extend(acc.drain());

                for (acc_size, acc_count) in &tmp {
                    for (child_size, child_count) in &child_data {
                        let combined_size = acc_size + child_size;
                        if combined_size > self.limit {
                            continue;
                        }
                        let combined_count = acc_count.to_owned() * child_count;
                        acc.entry(combined_size)
                            .and_modify(|c| *c += &combined_count)
                            .or_insert(combined_count);
                    }
                }

                tmp.clear();
                acc
            })
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if b.is_empty() {
            return DidMerge(false, false);
        }

        for (size, count) in b {
            a.entry(size)
                .and_modify(|c| {
                    *c += &count;
                })
                .or_insert(count);
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

        let data = ExprCount::new(10).one_shot_analysis(&egraph);
        let root_data: &HashMap<usize, BigUint> = &data[&egraph.find(apb)];

        assert_eq!(root_data[&5], 1usize.into());
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

        let data = ExprCount::new(10).one_shot_analysis(&egraph);

        let root_data: &HashMap<usize, BigUint> = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16usize.into());
    }

    #[test]
    fn halide_count_size() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(5).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();
        let egraph = eqsat.egraph();
        let root = eqsat.roots()[0];

        let data = ExprCount::new(16).one_shot_analysis(egraph);

        let root_data: &HashMap<usize, BigUint> = &data[&egraph.find(root)];

        assert_eq!(root_data[&16], 40512usize.into());
    }
}
