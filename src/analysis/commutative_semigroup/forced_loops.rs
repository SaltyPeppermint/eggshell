use std::collections::BTreeSet;
use std::fmt::Debug;
use std::sync::Arc;

use dashmap::DashMap;
use egg::DidMerge;
use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
use hashbrown::HashSet;

use super::CommutativeSemigroupAnalysis;
use super::Counter;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ENodeLoopData {
    loop_forcing: bool,
    transitive_closure: HashSet<Id>,
}

impl ENodeLoopData {
    fn new(eclass_id: Id) -> Self {
        Self {
            loop_forcing: false,
            transitive_closure: HashSet::from([eclass_id]),
        }
    }

    fn combine(&mut self, other: &ENodeLoopData) {
        self.loop_forcing = !self
            .transitive_closure
            .is_disjoint(&other.transitive_closure);
        self.transitive_closure.extend(&other.transitive_closure);
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EClassLoopData<L: Language>(HashMap<L, ENodeLoopData>);

impl<L: Language> EClassLoopData<L> {
    pub fn new(enode: L, transitive_closure: ENodeLoopData) -> Self {
        Self(HashMap::from([(enode, transitive_closure)]))
    }

    pub fn combine(&mut self, other: EClassLoopData<L>) {
        for (k, v_other) in other.0 {
            if let Some(v_self) = self.0.get_mut(&k) {
                v_self.combine(&v_other);
            } else {
                self.0.insert(k, v_other);
            }
        }
    }

    pub fn transitive_closure(&self) -> impl Iterator<Item = &ENodeLoopData> {
        self.0.iter().map(|(_, v)| v)
    }

    pub fn loop_forcing(&self) -> bool {
        self.0.iter().all(|(_, v)| v.loop_forcing)
    }

    pub fn get(&self, key: &L) -> Option<&ENodeLoopData> {
        self.0.get(key)
    }

    pub fn node_known_to_be_loop_forcing(&self, node: &L) -> bool {
        self.0.get(node).map(|n| n.loop_forcing).unwrap_or(false)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct LoopCause;

impl<L, N> CommutativeSemigroupAnalysis<L, N> for LoopCause
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    // Size and number of programs of that size
    type Data = EClassLoopData<L>;

    fn make(
        &self,
        _egraph: &EGraph<L, N>,
        eclass_id: Id,
        enode: &L,
        analysis_of: &Arc<DashMap<Id, Self::Data>>,
    ) -> Self::Data {
        // If any of the children loop, the node loops
        // Thankfully this is cached via `analysis_of`
        let transitive_closure =
            enode
                .children()
                .iter()
                .fold(ENodeLoopData::new(eclass_id), |acc, c_id| {
                    analysis_of
                        .get(c_id)
                        .unwrap()
                        .transitive_closure()
                        .fold(acc, |mut acc, v| {
                            acc.combine(v);
                            acc
                        })
                });
        EClassLoopData::new(enode.to_owned(), transitive_closure)
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if *a == b {
            return DidMerge(false, false);
        }
        a.combine(b);
        DidMerge(true, false)
    }
}

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
        analysis_of: &Arc<DashMap<Id, Self::Data>>,
    ) -> Self::Data {
        if self.loop_analysis[&eclass_id].node_known_to_be_loop_forcing(enode) {
            return None;
        }
        let mut children_data = Vec::new();
        for child_id in enode.children() {
            children_data.push(analysis_of.get(child_id).unwrap().clone());
        }

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

    use crate::eqsat::{self, EqsatConf};
    use crate::rewrite_system::halide::HalideLang;
    use crate::rewrite_system::{Halide, RewriteSystem};

    use super::*;

    #[test]
    fn simple_term_loop_causet() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let ma = egraph.add(SymbolLang::new("-", vec![a]));

        egraph.union(a, ma);
        egraph.rebuild();

        let data = LoopCause.one_shot_analysis(&egraph);
        let root_data = &data[&egraph.find(ma)];

        assert_eq!(
            root_data,
            &EClassLoopData::new(SymbolLang::leaf("a"), ENodeLoopData::new(a))
        );
    }

    #[test]
    fn slightly_complicated_loop_cause() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        egraph.union(b, apb);
        egraph.rebuild();

        let data = LoopCause.one_shot_analysis(&egraph);

        let root_data = &data[&egraph.find(apb)];
        assert_eq!(
            root_data,
            &EClassLoopData::new(SymbolLang::leaf("a"), ENodeLoopData::new(a))
        );
    }

    #[test]
    fn halide_loop_cause() {
        let start_expr: RecExpr<HalideLang> =
            "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
                .parse()
                .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(5).build();

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

        let data = LoopCause.one_shot_analysis(egraph);

        let root = egraph.find(eqsat.roots()[0]);
        let k = &start_expr[Id::from(0)];
        let root_data = data[&root].get(k).unwrap();

        assert_eq!(root_data, &ENodeLoopData::new(root));
    }

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

        let loop_analysis = LoopCause.one_shot_analysis(&egraph);
        println!("LOOP ANALYSIS DONE");
        let data = LoopFreeCount::new(loop_analysis).one_shot_analysis(egraph);

        let d: usize = *data[&Id::from(11)].as_ref().unwrap().get(&23).unwrap();
        assert_eq!(d, 1232);
    }
}
