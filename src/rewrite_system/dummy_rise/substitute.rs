use egg::{Id, Language, RecExpr, Symbol};

use super::{DummyRiseLang, unwrap_symbol};

pub fn expr_contains(e: &[DummyRiseLang], var: Symbol) -> bool {
    for node in e {
        if node == &DummyRiseLang::Symbol(var) {
            return true;
        }
    }
    false
}

pub fn substitute_expr(
    var: Symbol,
    expr: &RecExpr<DummyRiseLang>,
    body: &RecExpr<DummyRiseLang>,
) -> RecExpr<DummyRiseLang> {
    struct Env<'a> {
        var: Symbol,
        expr: &'a RecExpr<DummyRiseLang>,
        body: &'a [DummyRiseLang],
        result: &'a mut RecExpr<DummyRiseLang>,
    }

    fn add_expr(to: &mut RecExpr<DummyRiseLang>, e: &[DummyRiseLang], id: Id) -> Id {
        let node = e[usize::from(id)]
            .clone()
            .map_children(|c_id| add_expr(to, e, c_id));
        to.add(node)
    }

    fn find_fresh_symbol(a: &[DummyRiseLang], b: &[DummyRiseLang]) -> Symbol {
        for i in 0..usize::MAX {
            let s = Symbol::from(format!("s{i}"));
            if !expr_contains(a, s) && !expr_contains(b, s) {
                return s;
            }
        }
        panic!("could not find fresh symbol")
    }

    fn replace_expr(e: &[DummyRiseLang], id: Id, v: Symbol, v2: Symbol) -> RecExpr<DummyRiseLang> {
        fn replace_add(
            to: &mut RecExpr<DummyRiseLang>,
            e: &[DummyRiseLang],
            id: Id,
            v: Symbol,
            v2: Symbol,
        ) -> Id {
            let node = e[usize::from(id)].clone();
            let result = if node == DummyRiseLang::Symbol(v) {
                DummyRiseLang::Symbol(v2)
            } else {
                node.map_children(|c_id| replace_add(to, e, c_id, v, v2))
            };
            to.add(result)
        }
        let mut result = RecExpr::default();
        replace_add(&mut result, e, id, v, v2);
        result
    }

    fn rec(index: usize, env: &mut Env) -> Id {
        match &env.body[index] {
            &DummyRiseLang::Var(v)
                if env.body[usize::from(v)] == DummyRiseLang::Symbol(env.var) =>
            {
                add_expr(
                    env.result,
                    env.expr.as_ref(),
                    Id::from(env.expr.as_ref().len() - 1),
                )
            }
            DummyRiseLang::Var(_) | DummyRiseLang::Symbol(_) | DummyRiseLang::Number(_) => {
                add_expr(env.result, env.body, Id::from(index))
            }
            &DummyRiseLang::Lambda([v, _])
                if env.body[usize::from(v)] == DummyRiseLang::Symbol(env.var) =>
            {
                add_expr(env.result, env.body, Id::from(index))
            }
            &DummyRiseLang::Lambda([v, e])
                if expr_contains(env.expr.as_ref(), unwrap_symbol(&env.body[usize::from(v)])) =>
            {
                let v2 = find_fresh_symbol(env.body, env.expr.as_ref());
                let e2 = replace_expr(env.body, e, unwrap_symbol(&env.body[usize::from(v)]), v2);
                let mut e3 = substitute_expr(env.var, env.expr, &e2);
                let ide3 = Id::from(e3.as_ref().len() - 1);
                let v2e3 = e3.add(DummyRiseLang::Symbol(v2));
                e3.add(DummyRiseLang::Lambda([v2e3, ide3]));
                add_expr(env.result, e3.as_ref(), Id::from(e3.as_ref().len() - 1))
            }
            &DummyRiseLang::Lambda([v, e]) => {
                let v2 = rec(usize::from(v), env);
                let e2 = rec(usize::from(e), env);
                env.result.add(DummyRiseLang::Lambda([v2, e2]))
            }
            &DummyRiseLang::App([f, e]) => {
                let f2 = rec(usize::from(f), env);
                let e2 = rec(usize::from(e), env);
                env.result.add(DummyRiseLang::App([f2, e2]))
            }
            node => panic!("did not expect {node:?}"),
        }
    }

    let mut result = RecExpr::default();
    rec(
        body.as_ref().len() - 1,
        &mut Env {
            var,
            expr,
            body: body.as_ref(),
            result: &mut result,
        },
    );
    result
}

// returns a new body where var becomes expr
// pub fn substitute_eclass(egraph: &mut EGraph, var: Id, expr: Id, body: Id) -> Id {
//     unimplemented!();
/* DEPRECATED CODE
struct Env<'a> {
    egraph: &'a mut RiseEGraph,
    var: Id,
    expr: Id,
    visited: HashMap<Id, Id>
}

fn rec_class(eclass: Id, env: &mut Env) -> Id {
    match env.visited.get(&eclass).map(|id| *id) {
        Some(id) => id,
        None =>
            if !env.egraph[eclass].data.free.contains(&env.var) {
                eclass
            } else {
                let enodes = env.egraph[eclass].nodes.clone();
                // add a dummy node to avoid cycles
                let dummy = env.egraph.reserve();
                // env.egraph.add(DummyRiseLang::Symbol(format!("_s_{}_{}_{}", eclass, env.var, env.expr).into()));
                env.visited.insert(eclass, dummy);
                let final_id = enodes.into_iter().fold(dummy, |current_id, enode| {
                    let new_id = match enode {
                        DummyRiseLang::Var(v) if env.egraph.find(v) == env.egraph.find(env.var) => env.expr,
                        // ignore dummies
                        // DummyRiseLang::Symbol(s) if s.as_str().starts_with("_") => dummy,
                        DummyRiseLang::Var(_) | DummyRiseLang::Symbol(_) => {
                            panic!("{:?}", enode);
                            // keeping same e-node would merge the new class with previous class
                        }
                        DummyRiseLang::Lambda([v, e]) =>
                            if env.egraph.find(v) == env.egraph.find(env.var) {
                                panic!("{:?}", v)
                                // keeping same e-node would merge the new class with previous class
                            } else if env.egraph[env.expr].data.free.contains(&v) {
                                env.egraph.rebuild();
                                env.egraph.dot().to_svg("/tmp/cap-avoid.svg").unwrap();
                                panic!("FIXME: capture avoid {:?} {:?}", env.egraph[v], env.egraph[env.var]);
                                let v2 = env.egraph.add(
                                    DummyRiseLang::Symbol(format!("subs_{}_{}_{}", eclass, env.var, env.expr).into()));
                                println!("capture avoid {}, {}, {}, {}, {}, {}", eclass, v2, v, e, env.var, env.expr);
                                let e2 = replace_eclass(env.egraph, v, v2, e);
                                let r = DummyRiseLang::Lambda([v2, rec_class(e2, env)]);
                                env.egraph.add(r)
                            } else {
                                let r = DummyRiseLang::Lambda([v, rec_class(e, env)]);
                                env.egraph.add(r)
                            },
                        DummyRiseLang::App([f, e]) => {
                            let r = DummyRiseLang::App([rec_class(f, env), rec_class(e, env)]);
                            env.egraph.add(r)
                        }
                        _ => panic!("did not expect {:?}", enode)
                    };
                    let (new_id, _) = env.egraph.union(current_id, new_id);
                    new_id
                });
                env.visited.insert(eclass, final_id);
                final_id
            }
    }
}

let visited = HashMap::new();
// egraph.rebuild();
// egraph.dot().to_svg(format!("/tmp/before_{}_{}_{}.svg", var, expr, body)).unwrap();
let r = rec_class(body, &mut Env { egraph, var, expr, visited });
// egraph.rebuild();
// egraph.dot().to_svg(format!("/tmp/after_{}_{}_{}.svg", var, expr, body)).unwrap();
r
*/
// }
