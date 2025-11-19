use core::panic;

use egg::{Applier, EGraph, Id, Language, Pattern, PatternAst, RecExpr, Subst, Symbol, Var};
use hashbrown::HashSet;

use crate::rewrite_system::rise::rules::{add, add_expr};

use super::super::lang::{AppType, RisePrimitives, RiseTypes};
use super::{Rise, RiseAnalysis};

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
    index: u32,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> NotFreeIn<A> {
    pub fn new(var: &str, index: u32, applier: A) -> Self {
        NotFreeIn {
            var: var.parse().unwrap(),
            index,
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

pub struct VectorizeScalaFun<A: Applier<Rise, RiseAnalysis>> {
    var: Var,
    size_var: Var,
    vectorized_var: Var,
    applier: A,
}

impl<A: Applier<Rise, RiseAnalysis>> VectorizeScalaFun<A> {
    pub fn new(var: &str, size_var: &str, vectorized_var: &str, applier: A) -> Self {
        VectorizeScalaFun {
            var: var.parse().unwrap(),
            size_var: size_var.parse().unwrap(),
            vectorized_var: vectorized_var.parse().unwrap(),
            applier,
        }
    }
}

impl<A: Applier<Rise, RiseAnalysis>> Applier<Rise, RiseAnalysis> for VectorizeScalaFun<A> {
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
        let n = extracted_to_u32(size_extracted);
        if let Some(vectorized_expr) = vec_type_of(extracted, n, HashSet::new(), extracted.root()) {
            let mut substitution = subst.clone();
            substitution.insert(self.vectorized_var, egraph.add_expr(&vectorized_expr));
            self.applier
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        } else {
            Vec::new()
        }
    }
}

fn extracted_to_u32(expr: &RecExpr<Rise>) -> i32 {
    if let Rise::Integer(i) = expr[0.into()] {
        return i;
    }
    panic!("Unexpected thing in expr")
}

fn vec_type_of(expr: &RecExpr<Rise>, n: i32, v_env: HashSet<u32>, id: Id) -> Option<RecExpr<Rise>> {
    let Rise::TypeOf([e, ty]) = &expr[id] else {
        panic!("Not TypeOf! {:?}", expr[id]);
    };
    vec_expr(expr, n, v_env, *e, *ty)
}

fn vec_expr(
    expr: &RecExpr<Rise>,
    n: i32,
    v_env: HashSet<u32>,
    expr_id: Id,
    ty_id: Id,
) -> Option<RecExpr<Rise>> {
    match &expr[expr_id] {
        Rise::TypeOf(_) => panic!("unexpected TypeOf"),
        Rise::Nat(_) => panic!("unexpected Nat"),
        Rise::Type(_) => panic!("unexpected Type"),

        // Scala Code:
        // case Var(i) if vEnv(i) =>
        //   for { tv <- vecDT(expr.t, n, eg) }
        //     yield ExprWithHashCons(Var(i), tv)
        // case Var(_) => None
        Rise::Var(index) if v_env.contains(&index.value()) => Some(join_2_recexprs(
            RecExpr::from(vec![Rise::Var(*index)]),
            build(expr, ty_id),
            |in_new, out_new| Rise::TypeOf([in_new, out_new]),
        )),
        Rise::Var(_) => None,
        // Scala code:
        // case App(f, e) =>
        // for { fv <- vectorizeExpr(f, n, eg, vEnv); ev <- vectorizeExpr(e, n, eg, vEnv) }
        //   yield ExprWithHashCons(App(fv, ev), eg(fv.t).asInstanceOf[FunType[TypeId]].outT)
        Rise::App(app_ty, [f, e]) => {
            let fv = vec_type_of(expr, n, v_env.clone(), *f)?;
            let ev = vec_type_of(expr, n, v_env.clone(), *e)?;
            let output_ty_id = output_type(&fv, fv.root());

            let add_expr =
                join_2_recexprs(fv, ev, |fv_id, ev_id| Rise::App(*app_ty, [fv_id, ev_id]));
            Some(join_2_recexprs(
                add_expr,
                build(expr, output_ty_id),
                |app_id, new_ty_id| Rise::TypeOf([app_id, new_ty_id]),
            ))

            // todo!("eg(fv.t).asInstanceOf[FunType[TypeId]].outT");
        }

        // Scala code:
        // case Lambda(e) =>
        // for { ev <- vectorizeExpr(e, n, eg, vEnv.map(_ + 1) + 0);
        //       xtv <- vecDT(eg(expr.t).asInstanceOf[FunType[TypeId]].inT, n, eg) }
        //   yield ExprWithHashCons(Lambda(ev),  eg.add(FunType(xtv, ev.t)))
        Rise::Lambda(lam_ty, e) => {
            let mut v_env2 = v_env.into_iter().map(|i| i + 1).collect::<HashSet<_>>();
            v_env2.insert(0);
            let mut ev = vec_type_of(expr, n, v_env2, *e)?;
            ev.add(Rise::Lambda(*lam_ty, ev.root()));

            let input_type_id = input_type(expr, ty_id);
            let xtv = vec_type(expr, n, input_type_id)?;

            let Rise::TypeOf([_, evt_id]) = ev[ev.root()] else {
                panic!("This has to be an typeof");
            };
            let evt = build(&ev, evt_id);

            let fun_expr = join_2_recexprs(xtv, evt, |xtv_id, evt_id| {
                Rise::Type(RiseTypes::FunType([xtv_id, evt_id]))
            });

            Some(join_2_recexprs(ev, fun_expr, |ev_id, fun_id| {
                Rise::Type(RiseTypes::FunType([ev_id, fun_id]))
            }))
        }
        // Scala code:
        // case Literal(_) | NatLiteral(_) | IndexLiteral(_, _) =>
        // for { tv <- vecDT(expr.t, n, eg) }
        //   yield ExprWithHashCons(App(
        //     ExprWithHashCons(Primitive(rcp.vectorFromScalar.primitive), eg.add(FunType(expr.t, tv))),
        //     expr), tv)
        Rise::Integer(_) => {
            let tv = vec_type(expr, n, ty_id)?;
            let mut nodes = Vec::new();

            let vec_t_id = add_expr(&mut nodes, &tv);

            let scalar_t_id = add_expr(&mut nodes, &build(expr, ty_id));

            let prim_id = add(
                &mut nodes,
                Rise::Primitive(RisePrimitives::VectorFromScalar),
            );
            let fun_id = add(
                &mut nodes,
                Rise::Type(RiseTypes::FunType([scalar_t_id, vec_t_id])),
            );
            let t_of_fun_id = add(&mut nodes, Rise::TypeOf([prim_id, fun_id]));
            let orig_id = add_expr(&mut nodes, &build(expr, expr_id));
            let t_of_app_id = add(&mut nodes, Rise::App(AppType::App, [t_of_fun_id, orig_id]));

            nodes.push(Rise::TypeOf([t_of_app_id, vec_t_id]));

            Some(RecExpr::from(nodes))
        }

        // Scala code:
        // case Primitive(rcp.add() | rcp.mul() | rcp.fst() | rcp.snd()) =>
        // for { tv <- vecT(expr.t, n, eg) }
        //   yield ExprWithHashCons(expr.node, tv)
        // case Primitive(_) => None
        Rise::Primitive(p) => match p {
            RisePrimitives::Snd
            | RisePrimitives::Fst
            | RisePrimitives::Add
            | RisePrimitives::Mul => Some(join_2_recexprs(
                RecExpr::from(vec![expr[expr_id].clone()]),
                build(expr, ty_id),
                |e_new_id, t_new_id| Rise::TypeOf([e_new_id, t_new_id]),
            )),
            _ => None,
        },
    }

    //   case Composition(f, g) =>
    //     for { fv <- vectorizeExpr(f, n, eg, vEnv); gv <- vectorizeExpr(g, n, eg, vEnv) }
    //       yield ExprWithHashCons(Composition(fv, gv), eg.add(FunType(
    //         eg(fv.t).asInstanceOf[FunType[TypeId]].inT,
    //         eg(gv.t).asInstanceOf[FunType[TypeId]].outT,
    //       )))
    // }
}

fn vec_type(rec_expr: &RecExpr<Rise>, n: i32, id: Id) -> Option<RecExpr<Rise>> {
    if let Rise::Var(_) = rec_expr[id] {
        return None;
    }
    let Rise::Type(rise_type) = rec_expr[id] else {
        panic!("Expected typeof, got {:?}", rec_expr[id]);
    };
    match rise_type {
        RiseTypes::F32 => {
            let mut vec_t = RecExpr::default();
            let scalar_ty = vec_t.add(Rise::Type(RiseTypes::F32));
            let width = vec_t.add(Rise::Integer(n));
            vec_t.add(Rise::Type(RiseTypes::VecType([width, scalar_ty])));
            Some(vec_t)
        }
        RiseTypes::NatType
        | RiseTypes::VecType(_)
        | RiseTypes::IndexType(_)
        | RiseTypes::ArrType(_) => None,
        RiseTypes::PairType([a, b]) => {
            let a2 = vec_type(rec_expr, n, a)?;
            let b2 = vec_type(rec_expr, n, b)?;

            let new_pair_type = join_2_recexprs(a2, b2, |a_new, b_new| {
                Rise::Type(RiseTypes::PairType([a_new, b_new]))
            });
            Some(new_pair_type)
        }
        RiseTypes::FunType([input_ty, output_ty]) => {
            let input_ty2 = vec_type(rec_expr, n, input_ty)?;
            let output_ty2 = vec_type(rec_expr, n, output_ty)?;
            let new_fun_type = join_2_recexprs(input_ty2, output_ty2, |in_new, out_new| {
                Rise::Type(RiseTypes::FunType([in_new, out_new]))
            });
            Some(new_fun_type)
        }
        RiseTypes::NatFunType(inner_id)
        | RiseTypes::DataFunType(inner_id)
        | RiseTypes::NatNatFunType(inner_id) => vec_type(rec_expr, n, inner_id),
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

fn output_type(rec_expr: &RecExpr<Rise>, fun_type_id: Id) -> Id {
    let node = &rec_expr[fun_type_id];
    match node {
        Rise::TypeOf([_, ty]) => output_type(rec_expr, *ty),
        Rise::Type(ty) => match ty {
            RiseTypes::FunType([_, output_ty]) => *output_ty,
            RiseTypes::NatFunType(inner)
            | RiseTypes::DataFunType(inner)
            | RiseTypes::NatNatFunType(inner) => output_type(rec_expr, *inner),
            _ => panic!("Output of fun not found {node:?}"),
        },
        _ => panic!("Output of fun not found {node:?}"),
    }
}

fn input_type(rec_expr: &RecExpr<Rise>, fun_type_id: Id) -> Id {
    let node = &rec_expr[fun_type_id];
    match node {
        Rise::TypeOf([_, ty]) => input_type(rec_expr, *ty),
        Rise::Type(ty) => match ty {
            RiseTypes::FunType([input_ty, _]) => *input_ty,
            RiseTypes::NatFunType(inner)
            | RiseTypes::DataFunType(inner)
            | RiseTypes::NatNatFunType(inner) => input_type(rec_expr, *inner),
            _ => panic!("Input of fun not found {node:?}"),
        },
        _ => panic!("Input of fun not found {node:?}"),
    }
}

fn build(rec_expr: &RecExpr<Rise>, id: Id) -> RecExpr<Rise> {
    rec_expr[id]
        .clone()
        .build_recexpr(|child_id| rec_expr[child_id].clone())
}

fn join_2_recexprs<L: Language, F>(r1: RecExpr<L>, r2: RecExpr<L>, new: F) -> RecExpr<L>
where
    F: Fn(Id, Id) -> L,
{
    let mut nodes = Vec::new();
    let len_r1 = r1.as_ref().len();
    let len_r2 = r2.as_ref().len();
    nodes.extend(r1);
    nodes.extend(
        r2.into_iter()
            .map(|n| n.map_children(|i| Id::from(usize::from(i) + len_r1))),
    );
    nodes.push(new(Id::from(len_r1), Id::from(len_r1 + len_r2)));
    RecExpr::from(nodes)
}

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
