use std::cmp::Ordering;
use std::fmt::Debug;

use egg::{Analysis, AstSize, CostFunction, DidMerge, EGraph, Id, Language};
use rustc_hash::{FxHashMap, FxHashSet};

use super::hashcons::ExprHashCons;

pub trait SemiLatticeAnalysis<L: Language, N: Analysis<L>> {
    type Data: Debug;

    fn make<'a>(
        &mut self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
        Self: 'a;

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge;
}

pub fn one_shot_analysis<L: Language, N: Analysis<L>, B: SemiLatticeAnalysis<L, N>>(
    egraph: &EGraph<L, N>,
    analysis: &mut B,
    data: &mut FxHashMap<Id, B::Data>,
) {
    assert!(egraph.clean);

    let mut analysis_pending = FxHashSetQueuePop::<(L, Id)>::new();
    // works with queue but IndexSet is stack
    // IndexSet::<(L, Id)>::default();

    for eclass in egraph.classes() {
        for enode in &eclass.nodes {
            if enode.all(|c| data.contains_key(&egraph.find(c))) {
                analysis_pending.insert((enode.clone(), eclass.id));
            }
        }
    }

    resolve_pending_analysis(egraph, analysis, data, &mut analysis_pending);

    debug_assert!(egraph.classes().all(|eclass| data.contains_key(&eclass.id)));
}

fn resolve_pending_analysis<L: Language, N: Analysis<L>, B: SemiLatticeAnalysis<L, N>>(
    egraph: &EGraph<L, N>,
    analysis: &mut B,
    data: &mut FxHashMap<Id, B::Data>,
    analysis_pending: &mut FxHashSetQueuePop<(L, Id)>,
) {
    while let Some((node, id)) = analysis_pending.pop() {
        let u_node = node.clone().map_children(|id| egraph.find(id)); // find_mut?

        if u_node.all(|id| data.contains_key(&id)) {
            let cid = egraph.find(id); // find_mut?
            let eclass = &egraph[cid];
            let node_data = analysis.make(egraph, &u_node, &|id| &data[&id]);
            let new_data = match data.remove(&cid) {
                None => {
                    analysis_pending.extend(eclass.parents().map(|(n, id)| (n.clone(), id)));
                    node_data
                }
                Some(mut existing) => {
                    let DidMerge(may_not_be_existing, _) = analysis.merge(&mut existing, node_data);
                    if may_not_be_existing {
                        analysis_pending.extend(eclass.parents().map(|(n, id)| (n.clone(), id)));
                    }
                    existing
                }
            };
            data.insert(cid, new_data);
        } else {
            analysis_pending.insert((node, id));
        }
    }
}

pub struct FxHashSetQueuePop<T> {
    map: FxHashSet<T>,
    queue: std::collections::VecDeque<T>,
}

impl<T: Eq + std::hash::Hash + Clone> FxHashSetQueuePop<T> {
    pub fn new() -> Self {
        FxHashSetQueuePop {
            map: FxHashSet::default(),
            queue: std::collections::VecDeque::new(),
        }
    }

    pub fn insert(&mut self, t: T) {
        if self.map.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.map.remove(t));
        res
    }
}

impl<L: Language, N: Analysis<L>> SemiLatticeAnalysis<L, N> for AstSize {
    type Data = usize;

    fn make<'a>(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'a Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'a,
    {
        enode.fold(1usize, |size, id| size + analysis_of(id))
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

pub struct ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
{
    exprs: &'a mut ExprHashCons<L>,
    cost_fn: &'a mut CF,
    extracted: &'a FxHashMap<Id, (CF::Cost, Id)>,
}

impl<'a, L, CF> ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
{
    pub fn new(
        exprs: &'a mut ExprHashCons<L>,
        cost_fn: &'a mut CF,
        extracted: &'a FxHashMap<Id, (CF::Cost, Id)>,
    ) -> Self {
        Self {
            exprs,
            cost_fn,
            extracted,
        }
    }
}

impl<'a, L, N, CF> SemiLatticeAnalysis<L, N> for ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    type Data = Option<(CF::Cost, Id)>;

    fn make<'b>(
        &mut self,
        egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &impl Fn(Id) -> &'b Self::Data,
    ) -> Self::Data
    where
        Self::Data: 'b,
        CF::Cost: 'b,
    {
        let children_matching: Vec<_> = enode
            .children()
            .iter()
            .filter_map(|&c| {
                let data = (*analysis_of)(c);
                data.as_ref().map(|x| (c, x.clone()))
            })
            .collect();
        let children_any: Vec<_> = enode
            .children()
            .iter()
            .map(|&c| (c, self.extracted[&egraph.find(c)].clone()))
            .collect();

        let mut candidates = Vec::new();

        for (matching_child, matching) in &children_matching {
            let mut to_selected = FxHashMap::default();

            for (child, any) in &children_any {
                let selected = if child == matching_child {
                    matching
                } else {
                    any
                };
                to_selected.insert(child, selected);
            }

            candidates.push((
                self.cost_fn.cost(enode, |c| to_selected[&c].0.clone()),
                self.exprs
                    .add(enode.clone().map_children(|c| to_selected[&c].1)),
            ));
        }

        candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        //.map(|best| (id, best))
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let ord = match (&a, &b) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (&Some((ref ca, _)), &Some((ref cb, _))) => ca.cmp(cb),
        };
        match ord {
            Ordering::Equal => DidMerge(false, false),
            Ordering::Less => DidMerge(false, true),
            Ordering::Greater => {
                *a = b;
                DidMerge(true, false)
            }
        }
    }
}

pub(crate) struct ExtractAnalysis<'a, L, CF> {
    pub(crate) exprs: &'a mut ExprHashCons<L>,
    pub(crate) cost_fn: &'a mut CF,
}

impl<'a, L, CF> ExtractAnalysis<'a, L, CF> {
    pub(crate) fn new(exprs: &'a mut ExprHashCons<L>, cost_fn: &'a mut CF) -> Self {
        Self { exprs, cost_fn }
    }
}

impl<'a, L, N, CF> SemiLatticeAnalysis<L, N> for ExtractAnalysis<'a, L, CF>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
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

#[cfg(test)]
mod tests {
    use egg::SymbolLang;

    use super::*;

    #[test]
    fn simple_analysis() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let z = egraph.add(SymbolLang::leaf("0"));
        let apz = egraph.add(SymbolLang::new("+", vec![a, z]));

        egraph.union(a, apz);
        egraph.rebuild();

        let mut data = FxHashMap::default();
        one_shot_analysis(&egraph, &mut AstSize, &mut data);

        assert_eq!(data[&egraph.find(apz)], 1);
    }
}