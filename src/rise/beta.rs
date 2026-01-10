use egg::{Applier, EGraph, Id, Language, PatternAst, RecExpr, Subst, Symbol, Var};

use super::db::{Cutoff, Index, Shift};
use super::kind::Kind;
use super::kind::Kindable;
use super::shifted::shift_mut;
use super::{Rise, RiseAnalysis};

pub struct BetaExtractApplier {
    body: Var,
    subs: Var,
    kind: Kind,
}

impl BetaExtractApplier {
    pub fn new(body: &str, subs: &str, kind: Kind) -> Self {
        Self {
            body: body.parse().unwrap(),
            subs: subs.parse().unwrap(),
            kind,
        }
    }
}

impl Applier<Rise, RiseAnalysis> for BetaExtractApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Rise>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let Some(ex_body) = egraph[subst[self.body]].data.small_repr(egraph) else {
            return Vec::new();
        };
        let Some(ex_subs) = egraph[subst[self.subs]].data.small_repr(egraph) else {
            return Vec::new();
        };
        // println!("----");
        // println!("UNSHIFTED BODY:");
        // ex_body.pp(false);
        // println!("UNSHIFTED SUBSTITUTION:");
        // ex_subs.pp(false);

        let shifted = beta_reduce(&ex_body, ex_subs, self.kind);
        let id = egraph.add_expr(&shifted);
        egraph.union(eclass, id);
        Vec::new()
    }
}

pub fn beta_reduce(body: &RecExpr<Rise>, mut arg: RecExpr<Rise>, kind: Kind) -> RecExpr<Rise> {
    // println!("SHIFTED SUBSTITUTION:");
    shift_mut(&mut arg, Shift::up(kind), Cutoff::zero()); // shift up
    // arg2.pp(false);
    let mut new_body = replace(body, Index::zero(kind), arg);

    // println!("BODY BEFORE SHIFT DOWN WITH REPLACEMENT, CUTOFF 0, SHIFT: -1 with kind {kind}",);
    // body2.pp(false);
    shift_mut(&mut new_body, Shift::down(kind), Cutoff::zero()); // shift down
    // println!("SHIFTED BODY:");
    // body2.pp(false);
    new_body
}

fn replace(expr: &RecExpr<Rise>, to_replace: Index, mut subs: RecExpr<Rise>) -> RecExpr<Rise> {
    fn rec(
        result: &mut RecExpr<Rise>,
        expr: &RecExpr<Rise>,
        id: Id,
        to_replace: Index,
        subs: &mut RecExpr<Rise>,
    ) -> Id {
        match &expr[id] {
            Rise::Var(found) if to_replace == *found => super::add_expr(result, subs.clone()),
            Rise::Lambda(l, e) => {
                let kind = l.kind();
                shift_mut(subs, Shift::up(kind), Cutoff::zero());
                let e2 = rec(result, expr, *e, to_replace.inc(kind), subs);
                shift_mut(subs, Shift::down(kind), Cutoff::zero());
                result.add(Rise::Lambda(*l, e2))
            }

            // NatNatLam is not covered
            // Non-matching vars and lambdas are handled by the default case
            other => {
                let new_other = other
                    .clone()
                    .map_children(|i| rec(result, expr, i, to_replace, subs));
                result.add(new_other)
            }
        }
    }
    let mut result = RecExpr::default();
    rec(&mut result, expr, expr.root(), to_replace, &mut subs);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_reduce_applier() {
        fn check(body: &str, arg: &str, res: &str) {
            let b = &body.parse().unwrap();
            let a = arg.parse().unwrap();
            let r = res.parse().unwrap();
            assert_eq!(beta_reduce(b, a, Kind::Expr), r);
        }
        // (λ. (λ. ((λ. (0 1)) (0 1)))) --> (λ. (λ. ((0 1) 0)))
        // (λ. (0 1)) (0 1) --> (0 1) 0
        check("(app %e0 %e1)", "(app %e0 %e1)", "(app (app %e0 %e1) %e0)");
        // r1 = (app (lam (app "%e6" (app "%e5" "%e0"))) "%e0")
        // r2 = (app (lam (app "%e6" r1)) "%e0")
        // r3 = (app (lam (app "%e6" r2)) %e0)
        // (app map (lam (app "%e6" r3)))
        // --> (app map (lam (app "%e6" (app "%e5" (app "%e4" (app "%e3" (app "%e2" "%e0")))))))
        check("(app %e6 (app %e5 %e0))", "%e0", "(app %e5 (app %e4 %e0))");
        check(
            "(app %e6 (app %e5 (app %e4 %e0)))",
            "%e0",
            "(app %e5 (app %e4 (app %e3 %e0)))",
        );
        check(
            "(app %e6 (app %e5 (app %e4 (app %e3 %e0))))",
            "%e0",
            "(app %e5 (app %e4 (app %e3 (app %e2 %e0))))",
        );
    }
}
