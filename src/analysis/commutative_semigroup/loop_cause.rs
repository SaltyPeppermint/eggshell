use std::fmt::Debug;
use std::sync::Arc;
use std::sync::RwLock;

use egg::DidMerge;
use egg::{Analysis, EGraph, Id, Language};
use hashbrown::HashMap;
use hashbrown::HashSet;

use super::CommutativeSemigroupAnalysis;

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
        for l in &other.transitive_closure {
            self.loop_forcing = self.loop_forcing || !self.transitive_closure.insert(*l);
        }
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

    pub fn node_is_loop_forcing(&self, node: &L) -> bool {
        self.0[node].loop_forcing
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
        analysis_of: &Arc<RwLock<HashMap<Id, Self::Data>>>,
    ) -> Self::Data {
        // If any of the children loop, the node loops
        // Thankfully this is cached via `analysis_of`
        let transitive_closure =
            enode
                .children()
                .iter()
                .fold(ENodeLoopData::new(eclass_id), |acc, c_id| {
                    let a_o = analysis_of.read().unwrap();
                    a_o[c_id].transitive_closure().fold(acc, |mut acc, v| {
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

#[cfg(test)]
mod tests {
    use egg::{EGraph, RecExpr, SimpleScheduler, SymbolLang};

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
        let root = eqsat.roots()[0];

        let data = LoopCause.one_shot_analysis(egraph);

        let root_data = &data[&egraph.find(root)];

        assert_eq!(
            root_data,
            &EClassLoopData::new(HalideLang::Bool(true), ENodeLoopData::new(root))
        );
    }
}
