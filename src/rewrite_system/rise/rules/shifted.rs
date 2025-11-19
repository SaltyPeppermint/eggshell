use egg::{Applier, Id, Language, PatternAst, RecExpr, Subst, Symbol, Var};

use crate::rewrite_system::rise;

use super::{EGraph, Rise, RiseAnalysis, TypedIndex};

// pub fn shifted<A: Applier<Rise, RiseAnalysis>>(
//     var: &str,
//     shifted_var: &str,
//     shift: i32,
//     cutoff: u32,
//     applier: A,
// ) -> impl Applier<Rise, RiseAnalysis> {
//     Shifted {
//         var: var.parse().unwrap(),
//         new_var: shifted_var.parse().unwrap(),
//         shift,
//         cutoff: Index(cutoff),
//         applier,
//     }
// }

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: u32,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        Shifted {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift,
            cutoff,
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
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let shifted = shift_copy(extract, self.shift, self.cutoff);
        let mut substitution = subst.clone();
        substitution.insert(self.new_var, egraph.add_expr(&shifted));
        self.applier
            .apply_one(egraph, eclass, &substitution, searcher_ast, rule_name)
    }
}

// pub fn shifted_check<A: Applier<Rise, RiseAnalysis>>(
//     var: &str,
//     shifted_var: &str,
//     shift: i32,
//     cutoff: u32,
//     applier: A,
// ) -> impl Applier<Rise, RiseAnalysis> {
//     ShiftedCheck {
//         var: var.parse().unwrap(),
//         new_var: shifted_var.parse().unwrap(),
//         shift,
//         cutoff: Index(cutoff),
//         applier,
//     }
// }

pub struct ShiftedCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: u32,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ShiftedCheck<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        ShiftedCheck {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift,
            cutoff,
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

pub fn shift_copy(expr: &RecExpr<Rise>, shift: i32, cutoff: u32) -> RecExpr<Rise> {
    let mut result = expr.to_owned();
    shift_mut(&mut result, shift, cutoff);
    result
}

pub fn shift_mut(expr: &mut [Rise], shift: i32, cutoff: u32) {
    fn rec(expr: &mut [Rise], ei: usize, shift: i32, cutoff: u32) {
        match expr[ei] {
            Rise::Var(index) => {
                if index.value() >= cutoff {
                    let index2 = TypedIndex::new(
                        index.value().checked_add_signed(shift).unwrap(),
                        index.ty(),
                    );
                    expr[ei] = Rise::Var(index2);
                }
            }
            Rise::Lambda(_, e) => {
                rec(expr, usize::from(e), shift, cutoff + 1);
            }
            Rise::App(_, [f, e]) => {
                rec(expr, usize::from(f), shift, cutoff);
                rec(expr, usize::from(e), shift, cutoff);
            }
            Rise::TypeOf([e, t]) => {
                rec(expr, usize::from(e), shift, cutoff);
                rec(expr, usize::from(t), shift, cutoff);
            }
            Rise::Nat(rise_nat) => rise_nat.children().iter().for_each(|&id| {
                rec(expr, usize::from(id), shift, cutoff);
            }),
            Rise::Primitive(_) | Rise::Integer(_) => (),
            Rise::Type(rise_types) => rise_types.children().iter().for_each(|&id| {
                rec(expr, usize::from(id), shift, cutoff);
            }),
        }
    }
    rec(expr, expr.len() - 1, shift, cutoff);
}
