use egg::{Applier, Id, PatternAst, RecExpr, Subst, Symbol, Var};

use super::{EGraph, Index, Rise, RiseAnalysis};

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        Shifted {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift,
            cutoff: Index::new(cutoff),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for Shifted<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        // dbg!("oops we are bailing");
        // dbg!(self.var);
        // dbg!(subst);
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let shifted = shift_copy(extract, self.shift, self.cutoff);
        let mut substitution = subst.clone();
        substitution.insert(self.new_var, egraph.add_expr(&shifted));
        self.applier
            .apply_one(egraph, eclass, &substitution, searcher_ast, rule_name)
    }
}

pub struct ShiftedCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ShiftedCheck<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        ShiftedCheck {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift,
            cutoff: Index::new(cutoff),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for ShiftedCheck<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let shifted = shift_copy(extract, self.shift, self.cutoff);
        let expected = &egraph[subst[self.new_var]].data.beta_extract;
        if shifted == *expected {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

pub fn shift_copy(expr: &RecExpr<Rise>, shift: i32, cutoff: Index) -> RecExpr<Rise> {
    let mut result = expr.to_owned();
    shift_mut(&mut result, shift, cutoff);
    result
}

pub fn shift_mut(expr: &mut [Rise], shift: i32, cutoff: Index) {
    fn rec(expr: &mut [Rise], ei: usize, shift: i32, cutoff: Index) {
        match expr[ei] {
            Rise::Var(index) => {
                if index >= cutoff {
                    // let index2 = Index(index.0.checked_add_signed(shift).unwrap());
                    let index2 = index + shift;
                    expr[ei] = Rise::Var(index2);
                }
            } // TODO ALL THE OTHER LAMBDAS
            Rise::Lambda(e)
            | Rise::NatLambda(e)
            | Rise::DataLambda(e)
            | Rise::AddrLambda(e)
            | Rise::NatNatLambda(e) => {
                rec(expr, usize::from(e), shift, cutoff + 1);
            }
            Rise::App([f, e])
            | Rise::NatApp([f, e])
            | Rise::DataApp([f, e])
            | Rise::AddrApp([f, e])
            | Rise::NatNatApp([f, e]) => {
                rec(expr, usize::from(f), shift, cutoff);
                rec(expr, usize::from(e), shift, cutoff);
            }
            Rise::TypeOf([e, t]) => {
                rec(expr, usize::from(e), shift, cutoff);
                rec(expr, usize::from(t), shift, cutoff);
            }
            _ => (), // _ => x.children().iter().for_each(|id| {
                     //     rec(expr, usize::from(*id), shift, cutoff);
                     // }),
        }
    }
    rec(expr, expr.len() - 1, shift, cutoff);
}
