use std::fmt::Debug;

use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

use super::{Sketch, SketchLang};
use crate::analysis::semilattice::{
    ExtractAnalysis, ExtractContainsAnalysis, ExtractOnlyContainsAnalysis, SemiLatticeAnalysis,
};
use crate::utils::ExprHashCons;

/// Returns the best program satisfying `s` according to `cost_fn` that is represented in the `id` e-class of `egraph`, if it exists.
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
#[expect(clippy::too_many_lines)]
fn extract<L, N, CF>(
    sketch: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, N>,
) -> (ExprHashCons<L>, HashMap<Id, (CF::Cost, Id)>)
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: Ord + 'static,
{
    assert!(egraph.clean);
    let mut memo = HashMap::default();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = HashMap::default();
    let mut analysis = ExtractAnalysis::new(&mut exprs, &mut cost_fn);
    analysis.one_shot_analysis(egraph, &mut extracted);

    let res = rec_extract(
        sketch,
        sketch_root,
        &mut cost_fn,
        egraph,
        &mut exprs,
        &extracted,
        &mut memo,
    );

    (exprs, res)
}

/// Recursion for `eclass_extract`
fn rec_extract<L, A, CF>(
    sketch: &Sketch<L>,
    sketch_id: Id,
    cost_fn: &mut CF,
    egraph: &EGraph<L, A>,
    exprs: &mut ExprHashCons<L>,
    extracted: &HashMap<Id, (CF::Cost, Id)>,
    memo: &mut HashMap<Id, HashMap<Id, (CF::Cost, Id)>>,
) -> HashMap<Id, (CF::Cost, Id)>
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
        SketchLang::Any => extracted.clone(),
        SketchLang::Node(sketch_node) => {
            let children_matches = sketch_node
                .children()
                .iter()
                .map(|sid| rec_extract(sketch, *sid, cost_fn, egraph, exprs, extracted, memo))
                .collect::<Vec<_>>();

            if let Some(potential_ids) = egraph.classes_for_op(&sketch_node.discriminant()) {
                potential_ids
                    .flat_map(|id| {
                        let eclass = &egraph[id];
                        let mut candidates = Vec::new();

                        let mnode = &sketch_node.clone().map_children(|_| Id::from(0));
                        let _ = eclass.for_each_matching_node::<()>(mnode, |matched| {
                            let mut matches = Vec::new();
                            for (cm, id) in children_matches.iter().zip(matched.children()) {
                                if let Some(m) = cm.get(id) {
                                    matches.push(m);
                                } else {
                                    break;
                                }
                            }

                            if matches.len() == matched.len() {
                                let to_match = matched
                                    .children()
                                    .iter()
                                    .zip(matches.iter())
                                    .collect::<HashMap<_, _>>();
                                candidates.push((
                                    cost_fn.cost(matched, |c| to_match[&c].0.clone()),
                                    exprs.add(matched.clone().map_children(|c| to_match[&c].1)),
                                ));
                            }

                            Ok(())
                        });
                        candidates
                            .into_iter()
                            .min_by(|x, y| x.0.cmp(&y.0))
                            .map(|best| (id, best))
                    })
                    .collect()
            } else {
                HashMap::default()
            }
        }
        SketchLang::Contains(sid) => {
            let contained_matches =
                rec_extract(sketch, *sid, cost_fn, egraph, exprs, extracted, memo);

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matches.get(&eclass.id).cloned()))
                .collect::<HashMap<_, _>>();

            let mut analysis = ExtractContainsAnalysis::new(exprs, cost_fn, extracted);

            analysis.one_shot_analysis(egraph, &mut data);

            data.into_iter()
                .flat_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                .collect()
        }
        SketchLang::OnlyContains(sid) => {
            let contained_matches =
                rec_extract(sketch, *sid, cost_fn, egraph, exprs, extracted, memo);

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matches.get(&eclass.id).cloned()))
                .collect::<HashMap<_, _>>();

            let mut analysis: ExtractOnlyContainsAnalysis<'_, L, _> =
                ExtractOnlyContainsAnalysis::new(exprs, cost_fn);

            analysis.one_shot_analysis(egraph, &mut data);

            data.into_iter()
                .flat_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                .collect()
        }
        SketchLang::Or(sids) => {
            let matches = sids
                .iter()
                .map(|sid| rec_extract(sketch, *sid, cost_fn, egraph, exprs, extracted, memo))
                .collect::<Vec<_>>();
            let matching_ids = matches
                .iter()
                .flat_map(|m| m.keys())
                .collect::<HashSet<_>>();

            matching_ids
                .iter()
                .flat_map(|id| {
                    matches
                        .iter()
                        .flat_map(|ms| ms.get(*id))
                        .into_iter()
                        .min_by(|x, y| x.0.cmp(&y.0))
                        .map(|best| (**id, best.clone()))
                })
                .collect()
        }
    };

    memo.insert(sketch_id, result.clone());
    result
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr, SymbolLang};

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
    fn big_extract() {
        let expr_a = "(>> (lam (>> f1 (>> transpose transpose)) (lam (>> (>> f2 transpose) transpose) (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app iterateStream (var f4)) (app (lam x1 (app (app iterateStream (var f3)) (app (app map (lam mfu22 (app (var f2) (app (var f1) (var mfu22))))) (var x1)))) (var x2)))) (var x3))))))))) (>> (>> (>> transpose transpose) (>> (>> (>> (>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose))) transpose) (>> (>> (>> transpose transpose) (>> transpose transpose)) (>> (>> transpose transpose) transpose))) (>> (>> transpose transpose) (>> transpose transpose)))) (>> (>> transpose transpose) (>> transpose transpose))))".parse::<RecExpr<RiseLang>>().unwrap();

        let expr_b="(>> (lam (>> f1 (>> transpose transpose)) (lam (>> (>> f2 transpose) transpose) (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app iterateStream (var f4)) (app (lam x1 (app (app iterateStream (var f3)) (app (app map (lam mfu22 (app (var f2) (app (var f1) (var mfu22))))) (var x1)))) (var x2)))) (var x3))))))))) (>> (>> (>> transpose transpose) (>> transpose transpose)) (>> (>> transpose transpose) (>> transpose transpose))))".parse::<RecExpr<RiseLang>>().unwrap();

        let sketch = "(>> (lam (>> f1 (>> transpose transpose)) (lam (>> (>> f2 transpose) transpose) (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app iterateStream (var f4)) (app (lam x1 (app (app iterateStream (var f3)) (app (app map (lam ? (app (var f2) (app (var f1) (var ?))))) (var x1)))) (var x2)))) (var x3))))))))) (>> (>> (>> transpose transpose) (>> (>> (>> (>> (>> transpose transpose) (>> (>> transpose transpose) (>> transpose transpose))) transpose) (>> (>> (>> transpose transpose) (>> transpose transpose)) (>> (>> transpose transpose) transpose))) (>> (>> transpose transpose) (>> transpose transpose)))) (>> (>> transpose transpose) (>> transpose transpose))))".parse::<Sketch<RiseLang>>().unwrap();
        let mut egraph = EGraph::<RiseLang, ()>::default();
        let a_root = egraph.add_expr(&expr_a);
        let b_root = egraph.add_expr(&expr_b);

        egraph.union(a_root, b_root);
        egraph.rebuild();

        let (best_cost, best_expr) = eclass_extract(&sketch, AstSize, &egraph, a_root).unwrap();
        assert_eq!(best_cost, 108);
        assert_eq!(best_expr.to_string(), expr_a.to_string());
    }
}
