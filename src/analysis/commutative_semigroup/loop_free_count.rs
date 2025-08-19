use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::RwLock;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use crate::analysis::commutative_semigroup::loop_cause::EClassLoopData;

use super::CommutativeSemigroupAnalysis;
use super::Counter;

#[derive(Debug, Clone)]
pub struct LoopFreeCount<L: Language> {
    loop_analysis: HashMap<Id, EClassLoopData<L>>,
}

impl<L: Language> LoopFreeCount<L> {
    #[must_use]
    pub fn new(loop_analysis: HashMap<Id, EClassLoopData<L>>) -> Self {
        Self { loop_analysis }
    }
}

impl<L, N, C> CommutativeSemigroupAnalysis<L, N, C> for LoopFreeCount<L>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    // Size and number of programs of that size
    // If the node is loop-forcing, return None
    type Data = Option<HashMap<usize, C>>;

    fn make(
        &self,
        _egraph: &EGraph<L, N>,
        eclass_id: Id,
        enode: &L,
        analysis_of: &Arc<RwLock<HashMap<Id, Self::Data>>>,
    ) -> Self::Data {
        if self.loop_analysis[&eclass_id].node_is_loop_forcing(enode) {
            return None;
        }
        let mut children_data = Vec::new();
        let a_o = analysis_of.read().unwrap();
        for child_id in enode.children() {
            children_data.push(a_o[child_id].clone());
        }
        drop(a_o);

        let mut tmp = Vec::new();

        Some(children_data.into_iter().fold(
            HashMap::from([(1, C::one())]),
            |mut acc, child_data| {
                tmp.extend(acc.drain());

                for (acc_size, acc_count) in &tmp {
                    for (child_size, child_count) in child_data.iter().flatten() {
                        let combined_size = acc_size + child_size;
                        let combined_count = acc_count.to_owned() * child_count;
                        acc.entry(combined_size)
                            .and_modify(|c| *c += &combined_count)
                            .or_insert(combined_count);
                    }
                }

                tmp.clear();
                acc
            },
        ))
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let Some(b_inner) = b else {
            return DidMerge(false, false);
        };

        let Some(a_inner) = a else {
            *a = Some(b_inner);
            return DidMerge(true, false);
        };

        for (size, count) in b_inner {
            a_inner
                .entry(size)
                .and_modify(|c| {
                    *c += &count;
                })
                .or_insert(count);
        }
        DidMerge(true, false)
    }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, RecExpr, SimpleScheduler, SymbolLang};
    use hashbrown::HashSet;

    use crate::analysis::commutative_semigroup::LoopCause;
    use crate::eqsat::{self, EqsatConf};
    use crate::rewrite_system::halide::HalideLang;
    use crate::rewrite_system::{Halide, RewriteSystem};

    use super::*;

    #[test]
    fn simplest_loop_free_count() {
        let mut egraph = EGraph::<_, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        egraph.union(b, apb);
        egraph.rebuild();

        let loop_analysis = LoopCause.one_shot_analysis(&egraph);
        let data = LoopFreeCount::new(loop_analysis).one_shot_analysis(&egraph);

        let root_data: &Option<HashMap<_, usize>> = &data[&egraph.find(apb)];
        assert_eq!(root_data, &None);
    }

    #[test]
    fn simple_loop_free_count() {
        let mut egraph = EGraph::<_, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let atb = egraph.add(SymbolLang::new("*", vec![a, b]));
        let ma = egraph.add(SymbolLang::new("-", vec![a]));
        let pb = egraph.add(SymbolLang::new("+", vec![b]));

        egraph.union(atb, ma);
        egraph.rebuild();
        egraph.union(atb, a);
        egraph.rebuild();
        egraph.union(ma, pb);
        egraph.rebuild();

        let loop_analysis = LoopCause.one_shot_analysis(&egraph);
        let data = LoopFreeCount::new(loop_analysis).one_shot_analysis(&egraph);

        let root_data: &Option<HashMap<_, usize>> = &data[&egraph.find(atb)];
        assert_eq!(root_data, &None);
    }

    #[test]
    fn slightly_complicated_loop_free_count() {
        let mut egraph = EGraph::<_, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let c = egraph.add(SymbolLang::leaf("c"));
        let f = egraph.add(SymbolLang::new("f", vec![a, b]));
        let g = egraph.add(SymbolLang::new("g", vec![c]));

        egraph.union(a, f);
        egraph.rebuild();
        egraph.union(a, g);
        egraph.rebuild();

        let loop_analysis = LoopCause.one_shot_analysis(&egraph);
        let data = LoopFreeCount::new(loop_analysis).one_shot_analysis(&egraph);

        let root_data: &Option<HashMap<_, usize>> = &data[&egraph.find(g)];
        assert_eq!(root_data, &None);
    }

    #[test]
    fn halide_loop_free() {
        let start_expr: RecExpr<HalideLang> =
            "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
                .parse()
                .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = eqsat::eqsat(
            eqsat_conf,
            (&start_expr).into(),
            rules.as_slice(),
            None,
            &[],
            SimpleScheduler,
        );

        let egraph = eqsat.egraph();
        let root = eqsat.roots()[0];

        let loop_analysis = LoopCause.one_shot_analysis(&egraph);
        println!("LOOP ANALYSIS DONE");
        let data = LoopFreeCount::new(loop_analysis).one_shot_analysis(egraph);

        let root_data: &Option<HashMap<_, usize>> = &data[&egraph.find(root)];
        assert_eq!(root_data, &None);
    }
}
