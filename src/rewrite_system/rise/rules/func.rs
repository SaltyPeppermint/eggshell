use egg::{Applier, EGraph, Id, Pattern, PatternAst, RecExpr, Subst, Symbol, Var};
use hashbrown::HashSet;

use super::{Index, Rise, RiseAnalysis};
use crate::rewrite_system::rise::{add_expr, build};

pub fn pat(pat: &str) -> impl Applier<Rise, RiseAnalysis> {
    pat.parse::<Pattern<Rise>>().unwrap()
}

// pub fn not_free_in<A>(var: &str, index: u32, applier: A) -> impl Applier<Rise, RiseAnalysis>
// where
//     A: Applier<Rise, RiseAnalysis>,
// {
//     NotFreeIn {
//         var: var.parse().unwrap(),
//         index: Index(index),
//         applier,
//     }
// }

pub struct NotFreeIn<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    index: Index,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> NotFreeIn<A> {
    pub fn new(var: &str, index: u32, applier: A) -> Self {
        NotFreeIn {
            var: var.parse().unwrap(),
            index: Index(index),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for NotFreeIn<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let free_in = egraph[subst[self.var]].data.free.contains(&self.index);
        if free_in {
            Vec::new()
        } else {
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        }
    }
}

// pub fn vectorize_scalar_fun<A: Applier<Rise, RiseAnalysis>>(
//     var: &str,
//     size_var: &str,
//     vectorized_var: &str,
//     applier: A,
// ) -> impl Applier<Rise, RiseAnalysis> {
//     VectorizeScalaFun {
//         var: var.parse().unwrap(),
//         size_var: size_var.parse().unwrap(),
//         vectorized_var: vectorized_var.parse().unwrap(),
//         applier,
//     }
// }

pub struct VectorizeScalarFun<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    size_var: Var,
    vectorized_var: Var,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> VectorizeScalarFun<A> {
    pub fn new(var: &str, size_var: &str, vectorized_var: &str, applier: A) -> Self {
        VectorizeScalarFun {
            var: var.parse().unwrap(),
            size_var: size_var.parse().unwrap(),
            vectorized_var: vectorized_var.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for VectorizeScalarFun<A> {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Rise, RiseAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<Rise>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let extracted = &egraph[subst[self.var]].data.beta_extract.clone();
        let size_extracted = &egraph[subst[self.size_var]].data.beta_extract.clone();
        let n = extracted_int(size_extracted);
        if let Some((vectorized_expr, _)) = vec_expr(extracted, n, HashSet::new(), extracted.root())
        {
            let mut substitution = subst.clone();
            substitution.insert(self.vectorized_var, egraph.add_expr(&vectorized_expr));
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn extracted_int(expr: &RecExpr<Rise>) -> i32 {
    if let Rise::Integer(i) = expr[0.into()] {
        return i;
    }
    panic!("Unexpected thing in expr")
}

fn vec_expr(
    expr: &RecExpr<Rise>,
    n: i32,
    v_env: HashSet<Index>,
    type_of_id: Id,
) -> Option<(RecExpr<Rise>, Id)> {
    let Rise::TypeOf([expr_id, ty_id]) = &expr[type_of_id] else {
        panic!("Not TypeOf! {:?}", expr[type_of_id]);
    };
    match &expr[*expr_id] {
        // Scala Code:
        // case Var(i) if vEnv(i) =>
        //     for { tv <- vecDT(expr.t, n, eg) }
        //     yield ExprWithHashCons(Var(i), tv)
        // case Var(_) => None
        Rise::Var(index) if v_env.contains(index) => {
            let mut new = RecExpr::default();
            let vec_ty_id = add_expr(&mut new, vec_ty(expr, n, *ty_id)?);
            let var_id = new.add(Rise::Var(*index));
            new.add(Rise::TypeOf([var_id, vec_ty_id]));
            Some((new, vec_ty_id))
        }
        // Scala code:
        // case App(f, e) =>
        // for { fv <- vectorizeExpr(f, n, eg, vEnv); ev <- vectorizeExpr(e, n, eg, vEnv) }
        //   yield ExprWithHashCons(App(fv, ev), eg(fv.t).asInstanceOf[FunType[TypeId]].outT)
        Rise::App([f, e]) => {
            let (fv, fv_ty_id) = vec_expr(expr, n, v_env.clone(), *f)?;
            let (ev, _) = vec_expr(expr, n, v_env.clone(), *e)?;

            let Rise::FunType([_, output_ty_id]) = fv[fv_ty_id] else {
                panic!("No Fun type wrapped in here: {:?}", &fv[fv_ty_id])
            };

            let mut new = RecExpr::default();
            let fv_id = add_expr(&mut new, fv);
            let ev_id = add_expr(&mut new, ev);

            let app_id = new.add(Rise::App([fv_id, ev_id]));

            // The index for fv is preserved since join_expr only appends the second arg
            new.add(Rise::TypeOf([app_id, output_ty_id]));
            Some((new, output_ty_id))
        }
        // Scala code:
        // case Lambda(e) =>
        // for { ev <- vectorizeExpr(e, n, eg, vEnv.map(_ + 1) + 0);
        //       xtv <- vecDT(eg(expr.t).asInstanceOf[FunType[TypeId]].inT, n, eg) }
        //   yield ExprWithHashCons(Lambda(ev),  eg.add(FunType(xtv, ev.t)))
        Rise::Lambda(e) => {
            let v_env2 = v_env
                .into_iter()
                .map(|i| i + 1)
                .chain([Index(0)])
                .collect::<HashSet<_>>();

            // Vectorize e
            let (ev, ev_ty_id) = vec_expr(expr, n, v_env2, *e)?;
            let mut new = RecExpr::default();
            let typed_ev_id = add_expr(&mut new, ev);

            // Get input type of the original expr and vectorize it (xtv in scala)
            let Rise::FunType([input_ty_id, _]) = expr[*ty_id] else {
                panic!("No Fun type wrapped in here: {:?}", &expr[*ty_id])
            };
            let vec_input_ty = vec_ty(expr, n, input_ty_id)?;

            // Wrap in a lambda and append to the existing recexpr
            let lam_id = new.add(Rise::Lambda(typed_ev_id));
            let vec_input_ty_id = add_expr(&mut new, vec_input_ty);

            // Add the fun type of the new lam and wrap it in a typeof
            // evt index is valid since add_expr only appends
            let fun_id = new.add(Rise::FunType([vec_input_ty_id, ev_ty_id]));
            new.add(Rise::TypeOf([lam_id, fun_id]));

            Some((new, fun_id))
        }
        // Scala Code:
        // case Literal(_) | NatLiteral(_) | IndexLiteral(_, _) =>
        //     for { tv <- vecDT(expr.t, n, eg) }
        //     yield ExprWithHashCons(App(
        //         ExprWithHashCons(Primitive(rcp.vectorFromScalar.primitive), eg.add(FunType(expr.t, tv))),
        //         expr), tv)
        Rise::Integer(_) => {
            let vec_ty = vec_ty(expr, n, *ty_id)?;
            let mut new = RecExpr::default();

            let new_expr_id = add_expr(&mut new, build(expr, *expr_id));

            let new_ty_id = add_expr(&mut new, build(expr, *ty_id));
            let new_typed_expr_id = new.add(Rise::TypeOf([new_expr_id, new_ty_id]));

            let vec_ty_id = add_expr(&mut new, vec_ty);
            let prim_id = new.add(Rise::VectorFromScalar);
            let fun_id = new.add(Rise::FunType([new_ty_id, vec_ty_id]));
            let typed_prim_id = new.add(Rise::TypeOf([prim_id, fun_id]));

            let app_id = new.add(Rise::App([typed_prim_id, new_typed_expr_id]));
            let typed_app_id = new.add(Rise::TypeOf([app_id, vec_ty_id]));

            Some((new, typed_app_id))
        }
        // Scala Code:
        // case Primitive(rcp.add() | rcp.mul() | rcp.fst() | rcp.snd()) =>
        //   for { tv <- vecT(expr.t, n, eg) }
        //      yield ExprWithHashCons(expr.node, tv)
        Rise::Snd | Rise::Fst | Rise::Add | Rise::Mul => {
            let mut typed_prim = RecExpr::default();
            let vec_prim_ty_id = add_expr(&mut typed_prim, vec_ty(expr, n, *ty_id)?);
            let prim_id = typed_prim.add(expr[*expr_id].clone());
            typed_prim.add(Rise::TypeOf([prim_id, vec_prim_ty_id]));
            Some((typed_prim, vec_prim_ty_id))
        }
        Rise::Var(_)
        | Rise::NatApp(_)
        | Rise::DataApp(_)
        | Rise::AddrApp(_)
        | Rise::NatNatApp(_)
        | Rise::NatLambda(_)
        | Rise::DataLambda(_)
        | Rise::AddrLambda(_)
        | Rise::NatNatLambda(_) => None,
        other => panic!("Cannot vectorize this {other:?}"),
    }
    //   case Composition(f, g) =>
    //     for { fv <- vectorizeExpr(f, n, eg, vEnv); gv <- vectorizeExpr(g, n, eg, vEnv) }
    //       yield ExprWithHashCons(Composition(fv, gv), eg.add(FunType(
    //         eg(fv.t).asInstanceOf[FunType[TypeId]].inT,
    //         eg(gv.t).asInstanceOf[FunType[TypeId]].outT,
    //       )))
    // }
}

fn vec_ty(expr: &RecExpr<Rise>, n: i32, id: Id) -> Option<RecExpr<Rise>> {
    if let Rise::Var(_) = expr[id] {
        return None;
    }
    match expr[id] {
        Rise::F32 => {
            let mut vec_ty = RecExpr::default();
            let scalar_ty = vec_ty.add(Rise::F32);
            let width = vec_ty.add(Rise::Integer(n));
            vec_ty.add(Rise::VecType([width, scalar_ty]));
            Some(vec_ty)
        }
        Rise::Var(_) | Rise::NatType | Rise::VecType(_) | Rise::IndexType(_) | Rise::ArrType(_) => {
            None
        }
        Rise::PairType([a, b]) => {
            let vec_fst_ty = vec_ty(expr, n, a)?;
            let vec_snd_ty = vec_ty(expr, n, b)?;

            let mut vec_pair_ty = RecExpr::default();
            let vec_fst_ty_id = add_expr(&mut vec_pair_ty, vec_fst_ty);
            let vec_snd_ty_id = add_expr(&mut vec_pair_ty, vec_snd_ty);
            vec_pair_ty.add(Rise::PairType([vec_fst_ty_id, vec_snd_ty_id]));
            Some(vec_pair_ty)
        }
        Rise::FunType([input_ty, output_ty]) => {
            let vec_input_ty = vec_ty(expr, n, input_ty)?;
            let vec_output_ty = vec_ty(expr, n, output_ty)?;

            let mut vec_fun_ty = RecExpr::default();
            let vec_in_ty_id = add_expr(&mut vec_fun_ty, vec_input_ty);
            let vec_out_ty_id = add_expr(&mut vec_fun_ty, vec_output_ty);
            vec_fun_ty.add(Rise::FunType([vec_in_ty_id, vec_out_ty_id]));

            Some(vec_fun_ty)
        }
        _ => panic!("Cannot vectorize {:?}", expr[id]),
    }
}
// private def vecT(t: TypeId, n: NatId, eg: EGraph): Option[TypeId] = {
//     t match {
//       case dt @ DataTypeId(_) => vecDT(dt, n, eg)
//       case ndt @ NotDataTypeId(_) => eg(ndt) match {
//         case FunType(inT, outT) =>
//           for { inTV <- vecT(inT, n, eg); outTV <- vecT(outT, n, eg) }
//             yield eg.add(FunType(inTV, outTV))
//         case NatFunType(t) => ???
//         case DataFunType(t) => ???
//         case AddrFunType(t) => ???
//         case NatToNatFunType(t) => ???
//         case _: DataTypeNode[_, _] => throw new Exception("this should not happen")
//       }
//     }
//   }

//   private def vecDT(t: TypeId, n: NatId, eg: EGraph): Option[TypeId] = {
//     t match {
//       case dt @ DataTypeId(_) => vecDT(dt, n, eg)
//       case NotDataTypeId(_) => None
//     }
//   }

//   private def vecDT(t: DataTypeId, n: NatId, eg: EGraph): Option[DataTypeId] = {
//     eg(t) match {
//       case DataTypeVar(_) => None
//       case ScalarType(_) => Some(eg.add(VectorType(n, t)))
//       case NatType => None
//       case VectorType(_, _) => None
//       case IndexType(_) => None
//       case PairType(a, b) =>
//         for { a2 <- vecDT(a, n, eg); b2 <- vecDT(b, n, eg) }
//           yield eg.add(PairType(a2, b2))
//       case ArrayType(_, _) => None
//     }
//   }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pat_rule_1() {
        pat(
            "(typeOf (lam (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt0) ?dt3)) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3)))) (typeOf ?e0 (app (app fun ?dt0) ?dt3))) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3))) (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt4) ?dt0)) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0)))) (typeOf (lam (typeOf ?e2 ?dt5)) (app (app fun ?dt4) ?dt0))) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0))) (typeOf %0 (app (app arrT ?n1) ?dt4))) (app (app arrT ?n1) ?dt0))) (app (app arrT ?n1) ?dt3))) (app (app fun (app (app arrT ?n0) ?dt1)) (app (app arrT ?n0) ?dt2)))",
        );
    }
}
