use egg::{Applier, Id, PatternAst, RecExpr, Subst, Symbol, Var};

use super::{EGraph, Index, Rise, RiseAnalysis};

pub fn shifted<A>(
    var: &str,
    shifted_var: &str,
    shift: i32,
    cutoff: u32,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    Shifted {
        var: var.parse().unwrap(),
        new_var: shifted_var.parse().unwrap(),
        shift,
        cutoff: Index(cutoff),
        applier,
    }
}

struct Shifted<A> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: Index,
    applier: A,
}

impl<A> Applier<Rise, RiseAnalysis> for Shifted<A>
where
    A: Applier<Rise, RiseAnalysis>,
{
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

pub fn shifted_check<A>(
    var: &str,
    shifted_var: &str,
    shift: i32,
    cutoff: u32,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    ShiftedCheck {
        var: var.parse().unwrap(),
        new_var: shifted_var.parse().unwrap(),
        shift,
        cutoff: Index(cutoff),
        applier,
    }
}

struct ShiftedCheck<A> {
    var: Var,
    new_var: Var,
    shift: i32,
    cutoff: Index,
    applier: A,
}

impl<A> Applier<Rise, RiseAnalysis> for ShiftedCheck<A>
where
    A: Applier<Rise, RiseAnalysis>,
{
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
                    let index2 = Index(index.0.checked_add_signed(shift).unwrap());
                    expr[ei] = Rise::Var(index2);
                }
            }
            Rise::Lambda(e) => {
                rec(expr, usize::from(e), shift, Index(cutoff.0 + 1));
            }
            Rise::App([f, e]) => {
                rec(expr, usize::from(f), shift, cutoff);
                rec(expr, usize::from(e), shift, cutoff);
            }
            Rise::Symbol(_) => (),
            Rise::TypeOf(_)
            | Rise::Integer(_)
            | Rise::ArrType
            | Rise::VecType
            | Rise::PairType
            | Rise::IndexType
            | Rise::NatType
            | Rise::F32
            | Rise::ToMem
            | Rise::Split
            | Rise::Join
            | Rise::NatAdd(_)
            | Rise::NatSub(_)
            | Rise::NatMul(_)
            | Rise::NatDiv(_)
            | Rise::NatPow(_)
            | Rise::AsVector
            | Rise::AsScalar
            | Rise::Snd
            | Rise::Fst
            | Rise::Generate
            | Rise::Transpose
            | Rise::Unzip
            | Rise::Zip
            | Rise::MapPar
            | Rise::Reduce
            | Rise::ReduceSeq
            | Rise::ReduceSeqUnroll => unimplemented!(),
        }
    }
    rec(expr, expr.len() - 1, shift, cutoff);
}
