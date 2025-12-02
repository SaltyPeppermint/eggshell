use egg::{Applier, EGraph, Id, PatternAst, RecExpr, Subst, Symbol, Var};

use super::{Index, Kindable, Rise, RiseAnalysis, Shift};

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: Shift,
    cutoff: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(var_str: &str, shifted_var_str: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        let var: Var = var_str.parse().unwrap();
        let kind = var.kind().unwrap();
        Shifted {
            var,
            new_var: shifted_var_str.parse().unwrap(),
            shift: shift.try_into().unwrap(),
            cutoff: Index::new(cutoff, kind),
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

        // println!("Shifting by {} with cutoff {}", self.shift, self.cutoff,);
        // println!("Extracted:");
        // extract.pp(false);
        // println!("Shifted");
        // shifted.pp(false);

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
    pub fn new(var_str: &str, shifted_var_str: &str, shift: i32, cutoff: u32, applier: A) -> Self {
        let var: Var = var_str.parse().unwrap();
        let kind = var.kind().unwrap();
        ShiftedCheck {
            var,
            new_var: shifted_var_str.parse().unwrap(),
            shift: shift.try_into().unwrap(),
            cutoff: Index::new(cutoff, kind),
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
        // dbg!(extract);
        let shifted = shift_copy(extract, self.shift, self.cutoff);

        // dbg!(&shifted);
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
                if index >= cutoff && index.kind() == cutoff.kind() {
                    let index2 = index + shift;
                    expr[ei] = Rise::Var(index2);
                }
            }
            Rise::Lambda(e)
            | Rise::NatLambda(e)
            | Rise::DataLambda(e)
            | Rise::AddrLambda(e)
            | Rise::NatNatLambda(e) => {
                if expr[ei].kind() == cutoff.kind() {
                    rec(expr, e, shift, cutoff.inc());
                } else {
                    rec(expr, e, shift, cutoff);
                }
            }
            // Should be covered by others
            // Rise::App([f, e])
            // | Rise::NatApp([f, e])
            // | Rise::DataApp([f, e])
            // | Rise::AddrApp([f, e])
            // | Rise::NatNatApp([f, e]) => {
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

#[cfg(test)]
mod tests {
    use crate::rewrite_system::rise::kind::Kind;

    use super::*;

    #[test]
    fn shift_applier() {
        fn check(to_shift: &str, ground_truth: &str, cutoff: u32, shift: i32) {
            let a = &to_shift.parse().unwrap();
            let b = &ground_truth.parse().unwrap();
            let shifted = shift_copy(
                a,
                Shift::try_from(shift).unwrap(),
                Index::new(cutoff, Kind::Expr),
            );
            assert_eq!(&shifted, b);
        }
        check("(app %e0 %e1)", "(app %e1 %e2)", 0, 1);
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e1 %e2) f32)",
            0,
            1,
        );
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e0 %e2) f32)",
            1,
            1,
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e3) f32)) f32)",
            1,
            1,
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e1) f32)) f32)",
            1,
            -1,
        );
        check(
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) f32))",
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) f32))",
            0,
            2,
        );
    }
}
