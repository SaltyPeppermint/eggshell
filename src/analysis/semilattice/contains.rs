use std::cmp::Ordering;
use std::fmt::Debug;

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::SemiLatticeAnalysis;
use crate::utils::ExprHashCons;

#[derive(Debug)]
pub struct ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L> + Debug,
    CF::Cost: Debug,
{
    exprs: &'a mut ExprHashCons<L>,
    cost_fn: &'a mut CF,
    extracted: &'a HashMap<Id, (CF::Cost, Id)>,
}

impl<'a, L, CF> ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L> + Debug,
    CF::Cost: Debug,
{
    pub fn new(
        exprs: &'a mut ExprHashCons<L>,
        cost_fn: &'a mut CF,
        extracted: &'a HashMap<Id, (CF::Cost, Id)>,
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
    CF: CostFunction<L> + Debug,
    CF::Cost: Ord + Debug,
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
            let mut to_selected = HashMap::new();

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
