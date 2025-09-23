use std::{cmp::Ordering, fmt::Debug};

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::SemiLatticeAnalysis;
use crate::utils::ExprHashCons;

#[derive(Debug)]
pub(crate) struct ExtractAnalysis<'a, L: Language, CF> {
    pub(crate) exprs: &'a mut ExprHashCons<L>,
    pub(crate) cost_fn: &'a mut CF,
}

impl<'a, L: Language, CF> ExtractAnalysis<'a, L, CF> {
    pub(crate) fn new(exprs: &'a mut ExprHashCons<L>, cost_fn: &'a mut CF) -> Self {
        Self { exprs, cost_fn }
    }
}

impl<'a, L, A, CF> SemiLatticeAnalysis<L, A> for ExtractAnalysis<'a, L, CF>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'static,
{
    type Data = (CF::Cost, Id);

    fn make<'b>(
        &mut self,
        _egraph: &EGraph<L, A>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data
    where
        Self::Data: 'b,
    {
        let expr_node = enode.clone().map_children(|c| analysis_of[&c].1);
        let expr = self.exprs.add(expr_node);
        let cost = self.cost_fn.cost(enode, |c| analysis_of[&c].0.clone());
        (cost, expr)
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

#[derive(Debug)]
pub struct ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
{
    exprs: &'a mut ExprHashCons<L>,
    cost_fn: &'a mut CF,
    extracted: &'a HashMap<Id, (CF::Cost, Id)>,
}

impl<'a, L, CF> ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
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

impl<'a, L, A, CF> SemiLatticeAnalysis<L, A> for ExtractContainsAnalysis<'a, L, CF>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'static + Ord,
{
    type Data = Option<(CF::Cost, Id)>;

    fn make<'b>(
        &mut self,
        egraph: &EGraph<L, A>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data
    where
        Self::Data: 'b,
    {
        let candidates = push_extract_contains_candidates(
            self.exprs,
            self.cost_fn,
            self.extracted,
            egraph,
            enode,
            analysis_of,
        );
        candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        merge_best_option(a, b)
    }
}

pub(crate) fn push_extract_contains_candidates<L, A, CF>(
    exprs: &mut ExprHashCons<L>,
    cost_fn: &mut CF,
    extracted: &HashMap<Id, (CF::Cost, Id)>,
    egraph: &EGraph<L, A>,
    enode: &L,
    analysis_of: &HashMap<Id, Option<(CF::Cost, Id)>>,
) -> Vec<(<CF as CostFunction<L>>::Cost, Id)>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: 'static + Ord,
{
    // TODO: could probably simplify and optimize this code

    // Children that satisfy the sketch
    let children_matching: Vec<_> = {
        enode
            .children()
            .iter()
            .map(|&c| (c, &analysis_of[&c]))
            .collect()
    };

    // Children that satisfy '?'
    let children_any: Vec<_> = enode
        .children()
        .iter()
        .map(|&c| (c, extracted[&egraph.find(c)].clone()))
        .collect();

    let mut index_based_enode = enode.clone();
    for (index, id) in index_based_enode.children_mut().iter_mut().enumerate() {
        *id = Id::from(index);
    }

    let mut candidates = Vec::new();

    // If one child satisfies sketch, it's enough.
    // Take '?' options for the other children.
    // Accumulate all combination.
    for (matching_index, (_matching_id, matching_data)) in children_matching.iter().enumerate() {
        if let Some(matching_data) = matching_data {
            let to_selected = children_any
                .iter()
                .enumerate()
                .map(|(index, (_id, any_data))| {
                    let selected = if index == matching_index {
                        matching_data
                    } else {
                        any_data
                    };
                    (Id::from(index), selected)
                })
                .collect::<HashMap<_, _>>();

            let cost = cost_fn.cost(&index_based_enode, |c| to_selected[&c].0.clone());
            let new_expr = exprs.add(
                index_based_enode
                    .clone()
                    .map_children(|c| to_selected[&c].1),
            );
            candidates.push((cost, new_expr));
        }
    }
    candidates
}

#[derive(Debug)]
pub struct ExtractOnlyContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
{
    exprs: &'a mut ExprHashCons<L>,
    cost_fn: &'a mut CF,
}

impl<'a, L, CF> ExtractOnlyContainsAnalysis<'a, L, CF>
where
    L: Language,
    CF: CostFunction<L>,
{
    pub fn new(exprs: &'a mut ExprHashCons<L>, cost_fn: &'a mut CF) -> Self {
        Self { exprs, cost_fn }
    }
}

fn merge_best_option<Cost>(a: &mut Option<(Cost, Id)>, b: Option<(Cost, Id)>) -> DidMerge
where
    Cost: 'static + Ord,
{
    let ord = match (&a, &b) {
        (None, None) => Ordering::Equal,
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (Some((ca, _)), Some((cb, _))) => ca.cmp(cb),
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

impl<'a, L, A, CF> SemiLatticeAnalysis<L, A> for ExtractOnlyContainsAnalysis<'a, L, CF>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'static + Ord,
{
    type Data = Option<(CF::Cost, Id)>;

    fn make<'b>(
        &mut self,
        _egraph: &EGraph<L, A>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data
    where
        Self::Data: 'b,
    {
        if enode.is_leaf() {
            return None;
        }

        // Children that satisfy the sketch
        let children_matching = enode
            .children()
            .iter()
            .map(|&c| (c, &analysis_of[&c]))
            .collect::<HashMap<_, _>>();

        if children_matching.iter().any(|(_, data)| data.is_none()) {
            return None;
        };

        let cost = self
            .cost_fn
            .cost(enode, |c| children_matching[&c].as_ref().unwrap().0.clone());
        let new_exprs = self.exprs.add(
            enode
                .clone()
                .map_children(|c| children_matching[&c].as_ref().unwrap().1),
        );
        Some((cost, new_exprs))
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        merge_best_option(a, b)
    }
}
