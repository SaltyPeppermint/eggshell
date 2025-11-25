mod lang;
mod rules;

use egg::{
    Applier, AstSize, EGraph, ENodeOrVar, Extractor, Id, Language, Pattern, PatternAst, RecExpr,
    Runner, Searcher, Subst, Symbol, Var,
};
use hashbrown::HashSet;

use super::{Rise, RiseAnalysis};
pub use lang::Math;

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

fn simplify(nat_expr: &RecExpr<Math>) -> RecExpr<Rise> {
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
        let expected = lang::to_nat_expr(&egraph[subst[self.var]].data.beta_extract);
        let extracted = lang::to_nat_expr(&extract_small(egraph, &self.nat_pattern, subst));
        if check_equivalence(egraph.analysis.get_mut_term_bank(), (expected, extracted)) {
            println!("They are the same!");
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn check_equivalence<'a, 'b: 'a>(
    term_bank: &'b mut HashSet<(RecExpr<Math>, RecExpr<Math>)>,
    pair: (RecExpr<Math>, RecExpr<Math>),
) -> bool {
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
    if term_bank.contains(&pair) {
        println!("Found pair early");
        return true;
    }

    let runner = Runner::default()
        // .with_egraph(term_bank.clone())
        .with_expr(&pair.0)
        .with_expr(&pair.1)
        .with_hook(move |r| {
            if r.egraph.find(r.roots[0]) == r.egraph.find(r.roots[1]) {
                Err("HOOK".to_owned())
            } else {
                Ok(())
            }
        })
        .run(&rules::rules());
    if runner.egraph.find(runner.roots[0]) == runner.egraph.find(runner.roots[1]) {
        term_bank.insert(pair);
        return true;
    }
    false
}

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
