use egg::{Applier, EGraph, Id, Pattern, PatternAst, Searcher, Subst, Symbol, Var};

use super::{FreeBetaNatAnalysis, Rise};

pub struct ComputeNat<A: Applier<Rise, FreeBetaNatAnalysis>> {
    var: Var,
    nat_pattern: Pattern<Rise>,
    applier: A,
}

impl<A: Applier<Rise, FreeBetaNatAnalysis>> ComputeNat<A> {
    #[expect(dead_code)]
    pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
        ComputeNat {
            var: var.parse().unwrap(),
            nat_pattern: nat_pattern.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, FreeBetaNatAnalysis>> Applier<Rise, FreeBetaNatAnalysis> for ComputeNat<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, FreeBetaNatAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(_nat_matches) = self.nat_pattern.search_eclass(egraph, subst[self.var]) else {
            return vec![];
        };
        let expr = &egraph[subst[self.var]].data.beta_extract;
        let simplified_nat = super::try_simplify(expr).unwrap();
        let mut new_subst = subst.clone();
        let added_expr_id = egraph.add_expr(&simplified_nat);
        new_subst.insert(self.var, added_expr_id);

        self.applier
            .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
    }
}

pub struct ComputeNatCheck<A: Applier<Rise, FreeBetaNatAnalysis>> {
    var: Var,
    nat_pattern: Pattern<Rise>,
    applier: A,
}

impl<A: Applier<Rise, FreeBetaNatAnalysis>> ComputeNatCheck<A> {
    pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
        ComputeNatCheck {
            var: var.parse().unwrap(),
            nat_pattern: nat_pattern.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, FreeBetaNatAnalysis>> Applier<Rise, FreeBetaNatAnalysis>
    for ComputeNatCheck<A>
{
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, FreeBetaNatAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let expected = &egraph[subst[self.var]].data.beta_extract.clone();
        let extracted = &super::extract_small(egraph, &self.nat_pattern, subst);
        let a = &mut egraph.analysis;
        if super::check_equivalence(a, expected, extracted) {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}
