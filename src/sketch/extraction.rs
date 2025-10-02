use std::fmt::Debug;

use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;

use super::{Sketch, SketchLang};
use crate::analysis::semilattice::{
    ExtractAnalysis, ExtractContainsAnalysis, ExtractOnlyContainsAnalysis, SemiLatticeAnalysis,
};
use crate::utils::ExprHashCons;

/// Returns the best program satisfying `s` according to `cost_fn` that is represented in the `id` e-class of `egraph`, if it exists.
#[expect(clippy::missing_panics_doc)]
pub fn eclass_extract<L, A, CF>(
    sketch: &Sketch<L>,
    cost_fn: CF,
    egraph: &EGraph<L, A>,
    id: Id,
) -> Option<(CF::Cost, RecExpr<L>)>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'static + Ord,
{
    assert_eq!(egraph.find(id), id);
    let (exprs, eclass_to_best) = extract(sketch, cost_fn, egraph);
    eclass_to_best
        .get(&id)
        .map(|(best_cost, best_id)| (best_cost.clone(), exprs.extract(*best_id)))
}

/// Extract cheapest term matching the sketch
///
/// # Panics
///
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
#[expect(clippy::type_complexity)]
fn extract<L, N, CF>(
    sketch: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, N>,
) -> (ExprHashCons<L>, HashMap<Id, (CF::Cost, usize)>)
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: Ord + 'static,
{
    assert!(egraph.clean);
    let mut memo = HashMap::default();
    let mut exprs = ExprHashCons::new();

    let mut precomputed_any = HashMap::default();
    ExtractAnalysis::new(&mut exprs, &mut cost_fn).one_shot_analysis(egraph, &mut precomputed_any);

    let res = rec_extract(
        sketch,
        sketch.root(),
        &mut cost_fn,
        egraph,
        &mut exprs,
        &precomputed_any,
        &mut memo,
    );

    (exprs, res)
}

/// Recursion for `eclass_extract`
#[expect(clippy::too_many_lines)]
fn rec_extract<L, A, CF>(
    sketch: &Sketch<L>,
    sketch_id: Id,
    cost_fn: &mut CF,
    egraph: &EGraph<L, A>,
    exprs: &mut ExprHashCons<L>,
    precomputed_any: &HashMap<Id, (CF::Cost, usize)>,
    memo: &mut HashMap<Id, HashMap<Id, (CF::Cost, usize)>>,
) -> HashMap<Id, (CF::Cost, usize)>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: 'static + Ord,
{
    if let Some(value) = memo.get(&sketch_id) {
        return value.clone();
    }

    let result = match &sketch[sketch_id] {
        SketchLang::Any => precomputed_any.clone(),
        SketchLang::Node(inner_node) => {
            let children_matches = inner_node
                .children()
                .iter()
                .map(|sid| rec_extract(sketch, *sid, cost_fn, egraph, exprs, precomputed_any, memo))
                .collect::<Box<_>>();

            egraph
                // Get all the eclasses with a matching node
                .classes_for_op(&inner_node.discriminant())
                .map(|potential_ids| {
                    let mut node_to_child_indices = inner_node.clone();
                    node_to_child_indices
                        .children_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(child_index, id)| {
                            *id = Id::from(child_index);
                        });

                    potential_ids
                        .filter_map(|id| {
                            egraph[id]
                                .nodes
                                .iter()
                                // Matching does not care about children so we do not need to clone here
                                // This give us only the nodes in the eclass that match our inner_node
                                .filter(|node| node.matches(inner_node))
                                .filter_map(|matched_node| {
                                    // For each of the nodes in the eclass, we check if each child is viable
                                    let matches = children_matches
                                        .iter()
                                        .zip(matched_node.children())
                                        // If the child is viable, we get it's stats and put it into a map
                                        // from index_ids to data
                                        .map_while(|(cm, egraph_id)| cm.get(egraph_id))
                                        .collect::<Box<_>>();

                                    // Only if all children are viable, the potential eclass is an actual matching one!
                                    // We then calculate the cost for all combinations
                                    (matches.len() == matched_node.len()).then(|| {
                                        let cost = cost_fn.cost(&node_to_child_indices, |c| {
                                            matches[usize::from(c)].0.clone()
                                        });
                                        let added_id =
                                            exprs.add(node_to_child_indices.clone().map_children(
                                                |c| Id::from(matches[usize::from(c)].1),
                                            ));
                                        (cost, added_id)
                                    })
                                })
                                // We want the cheapest of all the nodes
                                .min_by(|x, y| x.0.cmp(&y.0))
                                .map(|best| (id, best))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        SketchLang::Contains(inner_sketch_id) => {
            let contained_matches = rec_extract(
                sketch,
                *inner_sketch_id,
                cost_fn,
                egraph,
                exprs,
                precomputed_any,
                memo,
            );

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matches.get(&eclass.id).cloned()))
                .collect();

            ExtractContainsAnalysis::new(exprs, cost_fn, precomputed_any)
                .one_shot_analysis(egraph, &mut data);

            data.into_iter()
                .filter_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                .collect()
        }
        SketchLang::OnlyContains(inner_sketch_id) => {
            let contained_matches = rec_extract(
                sketch,
                *inner_sketch_id,
                cost_fn,
                egraph,
                exprs,
                precomputed_any,
                memo,
            );

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matches.get(&eclass.id).cloned()))
                .collect();

            ExtractOnlyContainsAnalysis::new(exprs, cost_fn).one_shot_analysis(egraph, &mut data);

            data.into_iter()
                .filter_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                .collect()
        }
        SketchLang::Or(inner_sketch_ids) => {
            let matches = inner_sketch_ids
                .iter()
                .map(|sid| rec_extract(sketch, *sid, cost_fn, egraph, exprs, precomputed_any, memo))
                .collect::<Box<_>>();
            matches
                .iter()
                .flat_map(|m| m.keys())
                .filter_map(|id| {
                    matches
                        .iter()
                        .filter_map(|ms| ms.get(id))
                        .min_by(|x, y| x.0.cmp(&y.0))
                        .map(|best| (*id, best.clone()))
                })
                .collect()
        }
    };
    // DEBUG
    // if let SketchLang::Node(inner) = &sketch[sketch_id] {
    //     dbg!(inner);
    //     dbg!(&memo);
    // }
    memo.insert(sketch_id, result.clone());
    result
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr, SimpleScheduler, SymbolLang, rewrite};

    use crate::eqsat;
    use crate::eqsat::EqsatConf;
    use crate::rewrite_system::RewriteSystem;
    use crate::rewrite_system::Rise;
    use crate::rewrite_system::rise::RiseLang;
    use crate::sketch::contains;

    use super::*;

    #[test]
    fn simple_extract() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();
        let c_expr = "(h (g x))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add_expr(&a_expr);
        let b = egraph.add_expr(&b_expr);
        let c = egraph.add_expr(&c_expr);

        egraph.rebuild();

        let sat1 = contains(&sketch, &egraph);
        assert_eq!(sat1.len(), 5);
        assert!(sat1.contains(&a));
        assert!(sat1.contains(&b));
        assert!(!sat1.contains(&c));

        egraph.union(a, b);
        egraph.rebuild();

        let sat2 = contains(&sketch, &egraph);
        assert_eq!(sat2.len(), 4);
        assert!(sat2.contains(&a));
        assert!(sat2.contains(&egraph.find(b)));
        assert!(!sat2.contains(&c));

        let (best_cost, best_expr) = eclass_extract(&sketch, AstSize, &egraph, a).unwrap();
        assert_eq!(best_cost, 4);
        assert_eq!(best_expr, a_expr);
    }

    #[test]
    fn loop_extract() {
        let sketch = "(g (g (g ?)))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(f x)".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(g x)".parse::<RecExpr<SymbolLang>>().unwrap();
        let c_expr = "x".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add_expr(&a_expr);
        let b = egraph.add_expr(&b_expr);
        let c = egraph.add_expr(&c_expr);

        egraph.rebuild();
        egraph.union(a, b);
        egraph.union(b, c);
        egraph.rebuild();

        let root = egraph.find(a);

        let (best_cost, best_expr) = eclass_extract(&sketch, AstSize, &egraph, root).unwrap();
        assert_eq!(&best_expr.to_string(), "(g (g (g x)))");
        assert_eq!(best_cost, 4);
    }

    #[test]
    fn loop_extract2() {
        let sketch = "(f (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(f (f x))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "x".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add_expr(&a_expr);
        let b = egraph.add_expr(&b_expr);

        egraph.rebuild();
        egraph.union(a, b);
        egraph.rebuild();

        let root = egraph.find(a);

        let (best_cost, best_expr) = eclass_extract(&sketch, AstSize, &egraph, root).unwrap();
        assert_eq!(&best_expr.to_string(), "(f (f x))");
        assert_eq!(best_cost, 3);
    }

    #[test]
    fn big_extract() {
        let expr_a = "(>> (lam (>> f1 (>> transpose transpose)) (lam (>> (>> f2 transpose) transpose) (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app iterateStream (var f4)) (app (lam x1 (app (app iterateStream (var f3)) (app (app map (lam mfu22 (app (var f2) (app (var f1) (var mfu22))))) (var x1)))) (var x2)))) (var x3))))))))) (>> (>> (>> transpose transpose) (>> (>> (>> (>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose))) transpose) (>> (>> (>> transpose transpose) (>> transpose transpose)) (>> (>> transpose transpose) transpose))) (>> (>> transpose transpose) (>> transpose transpose)))) (>> (>> transpose transpose) (>> transpose transpose))))".parse::<RecExpr<RiseLang>>().unwrap();

        let sketch = "(>> (lam (>> f1 (>> transpose transpose)) (lam (>> (>> f2 transpose) transpose) (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app iterateStream (var f4)) (app (lam x1 (app (app iterateStream (var f3)) (app (app map (lam ? (app (var f2) (app (var f1) (var ?))))) (var x1)))) (var x2)))) (var x3))))))))) (>> (>> (>> transpose transpose) (>> (>> (>> (>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose))) transpose) (>> (>> (>> transpose transpose) (>> transpose transpose)) (>> (>> transpose transpose) transpose))) (>> (>> transpose transpose) (>> transpose transpose)))) (>> (>> transpose transpose) (>> transpose transpose))))".parse::<Sketch<RiseLang>>().unwrap();
        let mut egraph = EGraph::<RiseLang, ()>::default();
        let a_root = egraph.add_expr(&expr_a);

        egraph.rebuild();

        let (_, best_expr) = eclass_extract(&sketch, AstSize, &egraph, a_root).unwrap();
        assert_eq!(best_expr.to_string(), expr_a.to_string());

        let conf = EqsatConf::builder().iter_limit(1).build();
        let r = eqsat::eqsat(
            &conf,
            (&expr_a).into(),
            &Rise::full_rules(),
            None,
            SimpleScheduler,
        );
        let root = r.egraph().find(a_root);
        let (_, new_best_expr) = eclass_extract(&sketch, AstSize, r.egraph(), root).unwrap();
        assert_eq!(new_best_expr.to_string(), expr_a.to_string());
    }

    #[test]
    fn big_extract_2() {
        let expr_a =
            "(>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose)))"
                .parse::<RecExpr<SymbolLang>>()
                .unwrap();

        let sketch =
            "(>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose)))"
                .parse::<Sketch<SymbolLang>>()
                .unwrap();

        let mut egraph = EGraph::<_, ()>::default();
        let a_root = egraph.add_expr(&expr_a);

        egraph.rebuild();

        let (_, best_expr) = eclass_extract(&sketch, AstSize, &egraph, a_root).unwrap();
        assert_eq!(best_expr.to_string(), expr_a.to_string());

        let rules = vec![
            rewrite!("transpose-id-1";  "(>> (>> transpose transpose) ?x)" => "?x"),
            rewrite!("transpose-id-2";  "(>> ?x (>> transpose transpose))" => "?x"),
        ];

        let runner = egg::Runner::default()
            .with_scheduler(egg::SimpleScheduler)
            .with_iter_limit(1)
            .with_egraph(egraph)
            .run(&rules);
        let new_egraph = runner.egraph;
        let (_, new_best_expr) =
            eclass_extract(&sketch, AstSize, &new_egraph, new_egraph.find(a_root)).unwrap();
        assert_eq!(new_best_expr.to_string(), expr_a.to_string());
    }
}
