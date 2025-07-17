use egg::{Id, Language, RecExpr, Symbol};

use super::{RiseLang, unwrap_symbol};

pub fn expr_contains(e: &[RiseLang], var: Symbol) -> bool {
    for node in e {
        if node == &RiseLang::Symbol(var) {
            return true;
        }
    }
    false
}

pub fn substitute_expr(
    var: Symbol,
    expr: &RecExpr<RiseLang>,
    body: &RecExpr<RiseLang>,
) -> RecExpr<RiseLang> {
    struct Env<'a> {
        var: Symbol,
        expr: &'a RecExpr<RiseLang>,
        body: &'a [RiseLang],
        result: &'a mut RecExpr<RiseLang>,
    }

    fn add_expr(to: &mut RecExpr<RiseLang>, e: &[RiseLang], id: Id) -> Id {
        let node = e[usize::from(id)]
            .clone()
            .map_children(|c_id| add_expr(to, e, c_id));
        to.add(node)
    }

    fn find_fresh_symbol(a: &[RiseLang], b: &[RiseLang]) -> Symbol {
        for i in 0..usize::MAX {
            let s = Symbol::from(format!("s{i}"));
            if !expr_contains(a, s) && !expr_contains(b, s) {
                return s;
            }
        }
        panic!("could not find fresh symbol")
    }

    fn replace_expr(e: &[RiseLang], id: Id, v: Symbol, v2: Symbol) -> RecExpr<RiseLang> {
        fn replace_add(
            to: &mut RecExpr<RiseLang>,
            e: &[RiseLang],
            id: Id,
            v: Symbol,
            v2: Symbol,
        ) -> Id {
            let node = e[usize::from(id)].clone();
            let result = if node == RiseLang::Symbol(v) {
                RiseLang::Symbol(v2)
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
            &RiseLang::Var(v) if env.body[usize::from(v)] == RiseLang::Symbol(env.var) => add_expr(
                env.result,
                env.expr.as_ref(),
                Id::from(env.expr.as_ref().len() - 1),
            ),
            RiseLang::Var(_) | RiseLang::Symbol(_) | RiseLang::Number(_) => {
                add_expr(env.result, env.body, Id::from(index))
            }
            &RiseLang::Lambda([v, _]) if env.body[usize::from(v)] == RiseLang::Symbol(env.var) => {
                add_expr(env.result, env.body, Id::from(index))
            }
            &RiseLang::Lambda([v, e])
                if expr_contains(env.expr.as_ref(), unwrap_symbol(&env.body[usize::from(v)])) =>
            {
                let v2 = find_fresh_symbol(env.body, env.expr.as_ref());
                let e2 = replace_expr(env.body, e, unwrap_symbol(&env.body[usize::from(v)]), v2);
                let mut e3 = substitute_expr(env.var, env.expr, &e2);
                let ide3 = Id::from(e3.as_ref().len() - 1);
                let v2e3 = e3.add(RiseLang::Symbol(v2));
                e3.add(RiseLang::Lambda([v2e3, ide3]));
                add_expr(env.result, e3.as_ref(), Id::from(e3.as_ref().len() - 1))
            }
            &RiseLang::Lambda([v, e]) => {
                let v2 = rec(usize::from(v), env);
                let e2 = rec(usize::from(e), env);
                env.result.add(RiseLang::Lambda([v2, e2]))
            }
            &RiseLang::App([f, e]) => {
                let f2 = rec(usize::from(f), env);
                let e2 = rec(usize::from(e), env);
                env.result.add(RiseLang::App([f2, e2]))
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
                // env.egraph.add(RiseLang::Symbol(format!("_s_{}_{}_{}", eclass, env.var, env.expr).into()));
                env.visited.insert(eclass, dummy);
                let final_id = enodes.into_iter().fold(dummy, |current_id, enode| {
                    let new_id = match enode {
                        RiseLang::Var(v) if env.egraph.find(v) == env.egraph.find(env.var) => env.expr,
                        // ignore dummies
                        // RiseLang::Symbol(s) if s.as_str().starts_with("_") => dummy,
                        RiseLang::Var(_) | RiseLang::Symbol(_) => {
                            panic!("{:?}", enode);
                            // keeping same e-node would merge the new class with previous class
                        }
                        RiseLang::Lambda([v, e]) =>
                            if env.egraph.find(v) == env.egraph.find(env.var) {
                                panic!("{:?}", v)
                                // keeping same e-node would merge the new class with previous class
                            } else if env.egraph[env.expr].data.free.contains(&v) {
                                env.egraph.rebuild();
                                env.egraph.dot().to_svg("/tmp/cap-avoid.svg").unwrap();
                                panic!("FIXME: capture avoid {:?} {:?}", env.egraph[v], env.egraph[env.var]);
                                let v2 = env.egraph.add(
                                    RiseLang::Symbol(format!("subs_{}_{}_{}", eclass, env.var, env.expr).into()));
                                println!("capture avoid {}, {}, {}, {}, {}, {}", eclass, v2, v, e, env.var, env.expr);
                                let e2 = replace_eclass(env.egraph, v, v2, e);
                                let r = RiseLang::Lambda([v2, rec_class(e2, env)]);
                                env.egraph.add(r)
                            } else {
                                let r = RiseLang::Lambda([v, rec_class(e, env)]);
                                env.egraph.add(r)
                            },
                        RiseLang::App([f, e]) => {
                            let r = RiseLang::App([rec_class(f, env), rec_class(e, env)]);
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
