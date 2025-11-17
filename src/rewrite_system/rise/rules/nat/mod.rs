mod lang;
mod rules;

use egg::{
    Applier, AstSize, EGraph, Extractor, Id, Pattern, PatternAst, RecExpr, Runner, Searcher, Subst,
    Symbol, Var,
};

use lang::RiseMath;

use super::{Rise, RiseAnalysis};

// #[expect(dead_code)]
// pub fn compute_nat<A>(var: &str, nat_pattern: &str, applier: A) -> impl Applier<Rise, RiseAnalysis>
// where
//     A: Applier<Rise, RiseAnalysis>,
// {
//     ComputeNat {
//         var: var.parse().unwrap(),
//         nat_pattern: nat_pattern.parse().unwrap(),
//         applier,
//     }
// }

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
        let simplified_nat = simplify(&lang::to_nat_expr(expr));
        let mut substitution = subst.clone();
        substitution.insert(self.var, egraph.add_expr(&simplified_nat));
        self.applier
            .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
    }
}

fn simplify(nat_expr: &RecExpr<RiseMath>) -> RecExpr<Rise> {
    let rules = rules::rules();
    let runner = Runner::default().with_expr(nat_expr).run(&rules);
    let root = runner.roots.first().unwrap();
    let (_, expr) = Extractor::new(&runner.egraph, AstSize).find_best(*root);
    lang::to_rise_expr(&expr)
}

// pub fn compute_nat_check<A>(
//     var: &str,
//     nat_pattern: &str,
//     applier: A,
// ) -> impl Applier<Rise, RiseAnalysis>
// where
//     A: Applier<Rise, RiseAnalysis>,
// {
//     ComputeNatCheck {
//         var: var.parse().unwrap(),
//         nat_pattern: nat_pattern.parse().unwrap(),
//         applier,
//     }
// }

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
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let nat_pattern_extracted =
            super::extract_small(egraph, &self.nat_pattern, subst[self.var])
                .iter()
                .map(lang::to_nat_expr)
                .collect::<Box<[_]>>();
        if check_equivalence(&nat_pattern_extracted, &lang::to_nat_expr(extract)) {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn check_equivalence(
    nat_pattern_extracted: &[RecExpr<RiseMath>],
    expected: &RecExpr<RiseMath>,
) -> bool {
    let mut runner = Runner::default().with_expr(expected);
    for npe in nat_pattern_extracted {
        runner = runner.with_expr(npe);
    }
    runner = runner.run(&rules::rules());

    let (expected_root, npe_roots) = runner.roots.split_last().unwrap();
    let canonical_expected_root = runner.egraph.find(*expected_root);
    npe_roots
        .iter()
        .any(|npe_root| runner.egraph.find(*npe_root) == canonical_expected_root)
}
