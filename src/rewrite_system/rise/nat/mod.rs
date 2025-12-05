// pub mod cost;
// pub mod lang;
mod monomial;
mod polynomial;
// pub mod rules;

use egg::{
    Applier, EGraph, ENodeOrVar, Id, Language, Pattern, PatternAst, RecExpr, Searcher, Subst,
    Symbol, Var,
};

use crate::rewrite_system::rise::nat::polynomial::{PolyError, Polynomial};

use super::{Rise, RiseAnalysis};
use monomial::Monomial;

pub struct ComputeNat<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    nat_pattern: Pattern<Rise>,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ComputeNat<A> {
    #[expect(dead_code)]
    pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
        ComputeNat {
            var: var.parse().unwrap(),
            nat_pattern: nat_pattern.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ComputeNat<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(_nat_matches) = self.nat_pattern.search_eclass(egraph, subst[self.var]) else {
            return vec![];
        };
        let expr = &egraph[subst[self.var]].data.beta_extract;
        let simplified_nat = try_simplify(expr).unwrap();
        let mut new_subst = subst.clone();
        let added_expr_id = egraph.add_expr(&simplified_nat);
        new_subst.insert(self.var, added_expr_id);
        let mut ids = self
            .applier
            .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name);
        ids.push(added_expr_id);
        ids
    }
}

pub fn try_simplify(nat_expr: &RecExpr<Rise>) -> Result<RecExpr<Rise>, PolyError> {
    let mut polynomial: Polynomial = nat_expr.try_into()?;
    polynomial.simplify();
    Ok(polynomial.into())
}

pub struct ComputeNatCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    nat_pattern: Pattern<Rise>,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ComputeNatCheck<A> {
    pub fn new(var: &str, nat_pattern: &str, applier: A) -> Self {
        ComputeNatCheck {
            var: var.parse().unwrap(),
            nat_pattern: nat_pattern.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ComputeNatCheck<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let expected = &egraph[subst[self.var]].data.beta_extract.clone();
        let extracted = &extract_small(egraph, &self.nat_pattern, subst);
        let a = &mut egraph.analysis;
        if check_equivalence(a, expected, extracted) {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn check_equivalence<'a, 'b: 'a>(
    cache: &'b mut RiseAnalysis,
    lhs: &RecExpr<Rise>,
    rhs: &RecExpr<Rise>,
) -> bool {
    // check cache
    if let Some(equiv) = cache.check_cache_equiv(lhs, rhs) {
        return equiv;
    }

    let poly_lhs: Polynomial = lhs.try_into().unwrap();
    let poly_rhs: Polynomial = rhs.try_into().unwrap();

    if poly_lhs == poly_rhs {
        cache.add_pair_to_cache(lhs, rhs);
        return true;
    }
    false
}

// Quick check for trivial cases:
// fn quick_check(lhs: &RecExpr<Math>, lhs_id: Id, rhs: &RecExpr<Math>, rhs_id: Id) -> bool {
//     lhs[lhs_id].matches(&rhs[rhs_id])
//         && lhs[lhs_id]
//             .children()
//             .iter()
//             .zip(rhs[rhs_id].children())
//             .all(|(lcid, rcid)| quick_check(lhs, *lcid, rhs, *rcid))
// }

// if quick_check(expected, expected.root(), extracted, extracted.root()) {
//     return true;
// }

fn extract_small(
    egraph: &EGraph<Rise, RiseAnalysis>,
    pattern: &Pattern<Rise>,
    subst: &Subst,
) -> RecExpr<Rise> {
    fn rec(
        ast: &PatternAst<Rise>,
        id: Id,
        subst: &Subst,
        egraph: &EGraph<Rise, RiseAnalysis>,
    ) -> RecExpr<Rise> {
        match &ast[id] {
            ENodeOrVar::Var(w) => egraph[subst[*w]].data.beta_extract.clone(),
            ENodeOrVar::ENode(e) => {
                let new_e = e.clone();
                new_e.join_recexprs(|i| rec(ast, i, subst, egraph))
            }
        }
    }
    rec(&pattern.ast, pattern.ast.root(), subst, egraph)
}
