use egg::{Applier, EGraph, Id, PatternAst, RecExpr, Subst, Symbol, Var};

use super::indices::Shift;
use super::{Index, Rise, RiseAnalysis};

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: Shift,
    cutoff: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        Shifted {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift: shift.try_into().unwrap(),
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
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let shifted = shift_copy(extract, self.shift, self.cutoff);
        let mut new_subst = subst.clone();
        let added_expr_id = egraph.add_expr(&shifted);
        new_subst.insert(self.new_var, added_expr_id);
        let mut ids = self
            .applier
            .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name);
        ids.push(added_expr_id);
        ids
    }
}

pub struct ShiftedCheck<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: Shift,
    cutoff: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ShiftedCheck<A> {
    pub fn new(var: &str, shifted_var: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        ShiftedCheck {
            var: var.parse().unwrap(),
            new_var: shifted_var.parse().unwrap(),
            shift: shift.try_into().unwrap(),
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

pub fn shift_copy(expr: &RecExpr<Rise>, shift: Shift, cutoff: Index) -> RecExpr<Rise> {
    let mut result = expr.to_owned();
    shift_mut(&mut result, shift, cutoff);
    result
}

pub fn shift_mut(expr: &mut RecExpr<Rise>, shift: Shift, cutoff: Index) {
    fn rec(expr: &mut RecExpr<Rise>, ei: Id, shift: Shift, cutoff: Index) {
        // dbg!(&expr[ei]);
        // dbg!(&expr.len());
        match expr[ei] {
            Rise::Var(index) => {
                if index >= cutoff {
                    // let index2 = Index(index.0.checked_add_signed(shift).unwrap());
                    let index2 = index + shift;
                    expr[ei] = Rise::Var(index2);
                }
            }
            Rise::Lambda(e)
            | Rise::NatLambda(e)
            | Rise::DataLambda(e)
            | Rise::AddrLambda(e)
            | Rise::NatNatLambda(e) => {
                rec(expr, e, shift, cutoff.upshifted());
            }
            // Should be covered by the iter impl
            // Rise::App(ids)
            // | Rise::NatApp(ids)
            // | Rise::DataApp(ids)
            // | Rise::AddrApp(ids)
            // | Rise::NatNatApp(ids)=> {
            //     rec(expr, f, shift, cutoff);
            //     rec(expr, e, shift, cutoff);
            // }
            Rise::App(ids)
            | Rise::NatApp(ids)
            | Rise::DataApp(ids)
            | Rise::AddrApp(ids)
            | Rise::NatNatApp(ids)
            | Rise::TypeOf(ids)
            | Rise::FunType(ids)
            | Rise::ArrType(ids)
            | Rise::VecType(ids)
            | Rise::PairType(ids)
            | Rise::NatAdd(ids)
            | Rise::NatSub(ids)
            | Rise::NatMul(ids)
            | Rise::NatDiv(ids)
            | Rise::NatPow(ids) => {
                for id in ids {
                    rec(expr, id, shift, cutoff);
                }
            }
            Rise::IndexType(id)
            | Rise::NatFun(id)
            | Rise::DataFun(id)
            | Rise::AddrFun(id)
            | Rise::NatNatFun(id) => rec(expr, id, shift, cutoff),
            Rise::Let
            | Rise::NatType
            | Rise::F32
            | Rise::AsVector
            | Rise::AsScalar
            | Rise::VectorFromScalar
            | Rise::Snd
            | Rise::Fst
            | Rise::Add
            | Rise::Mul
            | Rise::ToMem
            | Rise::Split
            | Rise::Join
            | Rise::Generate
            | Rise::Transpose
            | Rise::Zip
            | Rise::Unzip
            | Rise::Map
            | Rise::MapPar
            | Rise::Reduce
            | Rise::ReduceSeq
            | Rise::ReduceSeqUnroll
            | Rise::Integer(_)
            | Rise::Float(_) => (),
        }
    }
    rec(expr, expr.root(), shift, cutoff);
}
