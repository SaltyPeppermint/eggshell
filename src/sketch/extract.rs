use std::mem::Discriminant;

use crate::{HashMap, HashSet};
use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};

use super::analysis;
use super::analysis::{ExtractAnalysis, ExtractContainsAnalysis, SatisfiesContainsAnalysis};
use super::hashcons::ExprHashCons;
use super::utils;
use super::{Sketch, SketchNode};

/// Is the `id` e-class of `egraph` representing at least one program satisfying `s`?
pub fn eclass_satisfies_sketch<L: Language, A: Analysis<L>>(
    sketch: &Sketch<L>,
    egraph: &EGraph<L, A>,
    id: Id,
) -> bool {
    satisfies_sketch(sketch, egraph).contains(&id)
}

/// Returns the set of e-classes of `egraph` that represent at least one program satisfying `s`.
/// # Panics
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
pub fn satisfies_sketch<L: Language, A: Analysis<L>>(
    sketch: &Sketch<L>,
    egraph: &EGraph<L, A>,
) -> HashSet<Id> {
    fn rec<L: Language, A: Analysis<L>>(
        sketch: &Sketch<L>,
        sketch_id: Id,
        egraph: &EGraph<L, A>,
        classes_by_op: &HashMap<Discriminant<L>, HashSet<Id>>,
        memo: &mut HashMap<Id, HashSet<Id>>,
    ) -> HashSet<Id> {
        if let Some(value) = memo.get(&sketch_id) {
            return value.clone();
        };

        let result = match &sketch[sketch_id] {
            SketchNode::Any => egraph.classes().map(|eclass| eclass.id).collect(),
            SketchNode::Node(node) => {
                let children_matches = node
                    .children()
                    .iter()
                    .map(|child_sketch_id| {
                        rec(sketch, *child_sketch_id, egraph, classes_by_op, memo)
                    })
                    .collect::<Vec<_>>();

                if let Some(potential_ids) = utils::classes_matching_op(node, classes_by_op) {
                    potential_ids
                        .iter()
                        .copied()
                        .filter(|&id| {
                            let eclass = &egraph[id];

                            let mnode = &node.clone().map_children(|_| Id::from(0));
                            utils::for_each_matching_node(eclass, mnode, |matched| {
                                let children_match = children_matches
                                    .iter()
                                    .zip(matched.children())
                                    .all(|(matches, child_id)| matches.contains(child_id));
                                if children_match {
                                    Err(())
                                } else {
                                    Ok(())
                                }
                            })
                            .is_err()
                        })
                        .collect()
                } else {
                    HashSet::default()
                }
            }
            SketchNode::Contains(inner_sketch_id) => {
                let contained_matched = rec(sketch, *inner_sketch_id, egraph, classes_by_op, memo);

                let mut data = egraph
                    .classes()
                    .map(|eclass| (eclass.id, contained_matched.contains(&eclass.id)))
                    .collect::<HashMap<_, bool>>();

                analysis::one_shot_analysis(egraph, &mut SatisfiesContainsAnalysis, &mut data);

                data.iter()
                    .filter_map(|(&id, &is_match)| if is_match { Some(id) } else { None })
                    .collect()
            }
            SketchNode::Or(inner_sketch_ids) => {
                let matches = inner_sketch_ids.iter().map(|inner_sketch_id| {
                    rec(sketch, *inner_sketch_id, egraph, classes_by_op, memo)
                });
                matches
                    .reduce(|a, b| a.union(&b).copied().collect())
                    .expect("empty or sketch")
            }
        };

        memo.insert(sketch_id, result.clone());
        result
    }

    assert!(egraph.clean);
    let mut memo = HashMap::<Id, HashSet<Id>>::default();
    // let sketch_nodes = s.as_ref();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    let classes_by_op = utils::new_classes_by_op(egraph);
    rec(sketch, sketch_root, egraph, &classes_by_op, &mut memo)
}

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
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    let (exprs, eclass_to_best) = extract_sketch(sketch, cost_fn, egraph);
    eclass_to_best
        .get(&id)
        .map(|(best_cost, best_id)| (best_cost.clone(), exprs.extract(*best_id)))
}

#[expect(
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]
fn extract_sketch<L, A, CF>(
    sketch: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, A>,
) -> (ExprHashCons<L>, HashMap<Id, (CF::Cost, Id)>)
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    fn rec<L, A, CF>(
        sketch: &Sketch<L>,
        sketch_id: Id,
        cost_fn: &mut CF,
        classes_by_op: &HashMap<Discriminant<L>, HashSet<Id>>,
        egraph: &EGraph<L, A>,
        exprs: &mut ExprHashCons<L>,
        extracted: &HashMap<Id, (CF::Cost, Id)>,
        memo: &mut HashMap<Id, HashMap<Id, (CF::Cost, Id)>>,
    ) -> HashMap<Id, (CF::Cost, Id)>
    where
        L: Language,
        A: Analysis<L>,
        CF: CostFunction<L>,
        CF::Cost: Ord,
    {
        if let Some(value) = memo.get(&sketch_id) {
            return value.clone();
        };

        let result = match &sketch[sketch_id] {
            SketchNode::Any => extracted.clone(),
            SketchNode::Node(node) => {
                let children_matches = node
                    .children()
                    .iter()
                    .map(|child_sketch_id| {
                        rec(
                            sketch,
                            *child_sketch_id,
                            cost_fn,
                            classes_by_op,
                            egraph,
                            exprs,
                            extracted,
                            memo,
                        )
                    })
                    .collect::<Vec<_>>();

                if let Some(potential_ids) = utils::classes_matching_op(node, classes_by_op) {
                    potential_ids
                        .iter()
                        .copied()
                        .filter_map(|id| {
                            let eclass = &egraph[id];
                            let mut candidates = Vec::new();

                            let mnode = &node.clone().map_children(|_| Id::from(0));
                            let _ = utils::for_each_matching_node(eclass, mnode, |matched| {
                                let mut matches = Vec::new();
                                for (cm, child_id) in
                                    children_matches.iter().zip(matched.children())
                                {
                                    if let Some(m) = cm.get(child_id) {
                                        matches.push(m);
                                    } else {
                                        break;
                                    }
                                }

                                if matches.len() == matched.len() {
                                    let to_match: HashMap<_, _> =
                                        matched.children().iter().zip(matches.iter()).collect();
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
            SketchNode::Contains(inner_sketch_id) => {
                let contained_matches = rec(
                    sketch,
                    *inner_sketch_id,
                    cost_fn,
                    classes_by_op,
                    egraph,
                    exprs,
                    extracted,
                    memo,
                );

                let mut data = egraph
                    .classes()
                    .map(|eclass| (eclass.id, contained_matches.get(&eclass.id).cloned()))
                    .collect::<HashMap<_, _>>();

                let mut analysis = ExtractContainsAnalysis::new(exprs, cost_fn, extracted);

                analysis::one_shot_analysis(egraph, &mut analysis, &mut data);

                data.into_iter()
                    .filter_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                    .collect()
            }
            SketchNode::Or(inner_sketch_ids) => {
                let matches = inner_sketch_ids
                    .iter()
                    .map(|inner_sketch_id| {
                        rec(
                            sketch,
                            *inner_sketch_id,
                            cost_fn,
                            classes_by_op,
                            egraph,
                            exprs,
                            extracted,
                            memo,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut matching_ids = HashSet::default();
                for m in &matches {
                    matching_ids.extend(m.keys());
                }

                matching_ids
                    .iter()
                    .filter_map(|id| {
                        let mut candidates = Vec::new();
                        for ms in &matches {
                            candidates.extend(ms.get(id));
                        }
                        candidates
                            .into_iter()
                            .min_by(|x, y| x.0.cmp(&y.0))
                            .map(|best| (*id, best.clone()))
                    })
                    .collect()
            }
        };

        memo.insert(sketch_id, result.clone());
        result
    }

    assert!(egraph.clean);
    let mut memo = HashMap::<Id, HashMap<Id, (CF::Cost, Id)>>::default();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = HashMap::default();
    let mut analysis = ExtractAnalysis {
        exprs: &mut exprs,
        cost_fn: &mut cost_fn,
    };
    let classes_by_op = utils::new_classes_by_op(egraph);
    analysis::one_shot_analysis(egraph, &mut analysis, &mut extracted);

    let res = rec(
        sketch,
        sketch_root,
        &mut cost_fn,
        &classes_by_op,
        egraph,
        &mut exprs,
        &extracted,
        &mut memo,
    );
    (exprs, res)
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr, SymbolLang};

    use super::*;

    #[test]
    fn simple_contains() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();
        let c_expr = "(h (g x))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add_expr(&a_expr);
        let b = egraph.add_expr(&b_expr);
        let c = egraph.add_expr(&c_expr);

        egraph.rebuild();

        let sat = satisfies_sketch(&sketch, &egraph);
        assert_eq!(sat.len(), 5);
        assert!(sat.contains(&a));
        assert!(sat.contains(&b));
        assert!(!sat.contains(&c));
    }

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
        egraph.union(a, b);
        egraph.rebuild();

        let sat = satisfies_sketch(&sketch, &egraph);
        assert_eq!(sat.len(), 4);
        assert!(sat.contains(&a));
        assert!(sat.contains(&egraph.find(b)));
        assert!(!sat.contains(&c));
    }

    #[test]
    fn simple_extract_cost() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let root_a = egraph.add_expr(&a_expr);
        let root_b = egraph.add_expr(&b_expr);

        egraph.rebuild();
        egraph.union(root_a, root_b);
        egraph.rebuild();

        let (best_cost, best_expr) = eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        assert_eq!(best_cost, 4);
        assert_eq!(best_expr, a_expr);
    }
}
