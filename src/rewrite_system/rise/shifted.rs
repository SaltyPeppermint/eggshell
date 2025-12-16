use egg::{Applier, EGraph, Id, PatternAst, RecExpr, Subst, Symbol, Var};

use super::{DBCutoff, DBShift, Kindable, Rise, RiseAnalysis};

pub struct Shifted<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    new_var: Var,
    shift: DBShift,
    cutoff: DBCutoff,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> Shifted<A> {
    pub fn new(
        var_str: &str,
        shifted_var_str: &str,
        shift: DBShift,
        cutoff: DBCutoff,
        applier: A,
    ) -> Self {
        Shifted {
            var: var_str.parse().unwrap(),
            new_var: shifted_var_str.parse().unwrap(),
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
    shift: DBShift,
    cutoff: DBCutoff,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> ShiftedCheck<A> {
    #[expect(unused)]
    pub fn new(
        var_str: &str,
        shifted_var_str: &str,
        shift: DBShift,
        cutoff: DBCutoff,
        applier: A,
    ) -> Self {
        ShiftedCheck {
            var: var_str.parse().unwrap(),
            new_var: shifted_var_str.parse().unwrap(),
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
        let expected = &egraph[subst[self.new_var]].data.beta_extract;
        let extract = &egraph[subst[self.var]].data.beta_extract;
        let shifted = shift_copy(extract, self.shift, self.cutoff);

        if shifted == *expected {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

pub fn shift_copy(expr: &RecExpr<Rise>, shift: DBShift, cutoff: DBCutoff) -> RecExpr<Rise> {
    let mut result = expr.to_owned();
    shift_mut(&mut result, shift, cutoff);
    result
}

pub fn shift_mut(expr: &mut RecExpr<Rise>, shift: DBShift, cutoff: DBCutoff) {
    fn rec(expr: &mut RecExpr<Rise>, id: Id, shift: DBShift, cutoff: DBCutoff) {
        // dbg!(&expr[ei]);
        // dbg!(&expr.len());
        match expr[id] {
            Rise::Var(index) => {
                if index.value() >= cutoff.of_index(index) {
                    let shifted_index = index + shift;
                    expr[id] = Rise::Var(shifted_index);
                }
            }
            Rise::Lambda(l, e) => {
                rec(expr, e, shift, cutoff.inc(l.kind()));
            }
            // Should be covered by others
            // Rise::App([f, e])
            // | Rise::NatApp([f, e])
            // | Rise::DataApp([f, e])
            // | Rise::AddrApp([f, e])
            // | Rise::NatNatApp([f, e]) => {
            //     rec(expr, f, shift, cutoff);
            //     rec(expr, e, shift, cutoff);
            Rise::App(_, c_ids)
            | Rise::TypeOf(c_ids)
            | Rise::FunType(c_ids)
            | Rise::ArrType(c_ids)
            | Rise::VecType(c_ids)
            | Rise::PairType(c_ids)
            | Rise::NatAdd(c_ids)
            | Rise::NatSub(c_ids)
            | Rise::NatMul(c_ids)
            | Rise::NatDiv(c_ids)
            | Rise::NatPow(c_ids) => {
                for c_id in c_ids {
                    rec(expr, c_id, shift, cutoff);
                }
            }
            Rise::NatFun(c_id)
            | Rise::DataFun(c_id)
            | Rise::AddrFun(c_id)
            | Rise::NatNatFun(c_id)
            | Rise::IndexType(c_id) => rec(expr, c_id, shift, cutoff),

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

    use super::*;

    #[test]
    fn shift_applier() {
        fn check(
            to_shift: &str,
            ground_truth: &str,
            cutoff: (u32, u32, u32, u32, u32),
            shift: (i32, i32, i32, i32, i32),
        ) {
            let a = &to_shift.parse().unwrap();
            let b = &ground_truth.parse().unwrap();
            let shifted = shift_copy(a, shift.into(), cutoff.into());
            assert_eq!(&shifted, b);
        }
        check(
            "(app %e0 %e1)",
            "(app %e1 %e2)",
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e1 %e2) f32)",
            (0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (app %e0 %e1) f32)",
            "(typeOf (app %e0 %e2) f32)",
            (1, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e3) f32)) f32)",
            (1, 0, 0, 0, 0),
            (1, 0, 0, 0, 0),
        );
        check(
            "(typeOf (lam (typeOf (app %e0 %e2) f32)) f32)",
            "(typeOf (lam (typeOf (app %e0 %e1) f32)) f32)",
            (1, 0, 0, 0, 0),
            (-1, 0, 0, 0, 0),
        );
        check(
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e3 (pairT f32 f32))) f32)) f32))",
            "(lam (typeOf (app (typeOf (app (typeOf mul (fun f32 (fun f32 f32))) (typeOf (app (typeOf fst (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) (fun f32 f32)) (typeOf (app (typeOf snd (fun (pairT f32 f32) f32)) (typeOf %e5 (pairT f32 f32))) f32)) f32))",
            (0, 0, 0, 0, 0),
            (2, 0, 0, 0, 0),
        );
    }
}
