use std::mem::Discriminant;

use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use rustc_hash::{FxHashMap, FxHashSet};

use super::analysis::{ExtractAnalysis, ExtractContainsAnalysis, SatisfiesContainsAnalysis};
use super::hashcons::ExprHashCons;
use super::utils;
use super::{Sketch, SketchNode};

/// Is the `id` e-class of `egraph` representing at least one program satisfying `s`?
pub fn eclass_satisfies_sketch<L: Language, A: Analysis<L>>(
    s: &Sketch<L>,
    egraph: &EGraph<L, A>,
    id: Id,
) -> bool {
    satisfies_sketch(s, egraph).contains(&id)
}

/// Returns the set of e-classes of `egraph` that represent at least one program satisfying `s`.
pub fn satisfies_sketch<L: Language, A: Analysis<L>>(
    s: &Sketch<L>,
    egraph: &EGraph<L, A>,
) -> FxHashSet<Id> {
    assert!(egraph.clean);
    let mut memo = FxHashMap::<Id, FxHashSet<Id>>::default();
    let sketch_nodes = s.as_ref();
    let sketch_root = Id::from(sketch_nodes.len() - 1);
    let classes_by_op = utils::new_classes_by_op(egraph);
    satisfies_sketch_rec(sketch_nodes, sketch_root, egraph, &classes_by_op, &mut memo)
}

fn satisfies_sketch_rec<L: Language, A: Analysis<L>>(
    s_nodes: &[SketchNode<L>],
    s_index: Id,
    egraph: &EGraph<L, A>,
    classes_by_op: &FxHashMap<Discriminant<L>, FxHashSet<Id>>,
    memo: &mut FxHashMap<Id, FxHashSet<Id>>,
) -> FxHashSet<Id> {
    if let Some(value) = memo.get(&s_index) {
        return value.clone();
    };

    let result = match &s_nodes[usize::from(s_index)] {
        SketchNode::Any => egraph.classes().map(|c| c.id).collect(),
        SketchNode::Node(node) => {
            let children_matches = node
                .children()
                .iter()
                .map(|sid| satisfies_sketch_rec(s_nodes, *sid, egraph, classes_by_op, memo))
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
                                .all(|(matches, id)| matches.contains(id));
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
                FxHashSet::default()
            }
        }
        SketchNode::Contains(sid) => {
            let contained_matched =
                satisfies_sketch_rec(s_nodes, *sid, egraph, classes_by_op, memo);

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matched.contains(&eclass.id)))
                .collect::<FxHashMap<_, bool>>();

            super::analysis::one_shot_analysis(egraph, &mut SatisfiesContainsAnalysis, &mut data);

            data.iter()
                .filter_map(|(&id, &is_match)| if is_match { Some(id) } else { None })
                .collect()
        }
        SketchNode::Or(sids) => {
            let matches = sids
                .iter()
                .map(|sid| satisfies_sketch_rec(s_nodes, *sid, egraph, classes_by_op, memo));
            matches
                .reduce(|a, b| a.union(&b).copied().collect())
                .expect("empty or sketch")
        }
    };

    memo.insert(s_index, result.clone());
    result
}

/// Returns the best program satisfying `s` according to `cost_fn` that is represented in the `id` e-class of `egraph`, if it exists.
pub fn eclass_extract_sketch<L, A, CF>(
    s: &Sketch<L>,
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
    let (exprs, eclass_to_best) = extract_sketch(s, cost_fn, egraph);
    eclass_to_best
        .get(&id)
        .map(|(best_cost, best_id)| (best_cost.clone(), exprs.extract(*best_id)))
}

#[allow(clippy::type_complexity)]
fn extract_sketch<L, A, CF>(
    s: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, A>,
) -> (ExprHashCons<L>, FxHashMap<Id, (CF::Cost, Id)>)
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    assert!(egraph.clean);
    let mut memo = FxHashMap::<Id, FxHashMap<Id, (CF::Cost, Id)>>::default();
    let sketch_nodes = s.as_ref();
    let sketch_root = Id::from(sketch_nodes.len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = FxHashMap::default();
    let mut analysis = ExtractAnalysis {
        exprs: &mut exprs,
        cost_fn: &mut cost_fn,
    };
    let classes_by_op = utils::new_classes_by_op(egraph);
    super::analysis::one_shot_analysis(egraph, &mut analysis, &mut extracted);

    let res = extract_sketch_rec(
        sketch_nodes,
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

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn extract_sketch_rec<L, A, CF>(
    s_nodes: &[SketchNode<L>],
    s_index: Id,
    cost_fn: &mut CF,
    classes_by_op: &FxHashMap<Discriminant<L>, FxHashSet<Id>>,
    egraph: &EGraph<L, A>,
    exprs: &mut ExprHashCons<L>,
    extracted: &FxHashMap<Id, (CF::Cost, Id)>,
    memo: &mut FxHashMap<Id, FxHashMap<Id, (CF::Cost, Id)>>,
) -> FxHashMap<Id, (CF::Cost, Id)>
where
    L: Language,
    A: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    if let Some(value) = memo.get(&s_index) {
        return value.clone();
    };

    let result = match &s_nodes[usize::from(s_index)] {
        SketchNode::Any => extracted.clone(),
        SketchNode::Node(node) => {
            let children_matches = node
                .children()
                .iter()
                .map(|sid| {
                    extract_sketch_rec(
                        s_nodes,
                        *sid,
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
                            for (cm, id) in children_matches.iter().zip(matched.children()) {
                                if let Some(m) = cm.get(id) {
                                    matches.push(m);
                                } else {
                                    break;
                                }
                            }

                            if matches.len() == matched.len() {
                                let to_match: FxHashMap<_, _> =
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
                FxHashMap::default()
            }
        }
        SketchNode::Contains(sid) => {
            let contained_matches = extract_sketch_rec(
                s_nodes,
                *sid,
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
                .collect::<FxHashMap<_, _>>();

            let mut analysis = ExtractContainsAnalysis::new(exprs, cost_fn, extracted);

            super::analysis::one_shot_analysis(egraph, &mut analysis, &mut data);

            data.into_iter()
                .filter_map(|(id, maybe_best)| maybe_best.map(|b| (id, b)))
                .collect()
        }
        SketchNode::Or(sids) => {
            let matches = sids
                .iter()
                .map(|sid| {
                    extract_sketch_rec(
                        s_nodes,
                        *sid,
                        cost_fn,
                        classes_by_op,
                        egraph,
                        exprs,
                        extracted,
                        memo,
                    )
                })
                .collect::<Vec<_>>();
            let mut matching_ids = FxHashSet::default();
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

    memo.insert(s_index, result.clone());
    result
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
        fn comp_eclass_extracted_sketch<L, A, CF>(
            s: &Sketch<L>,
            cost_fn_1: CF,
            cost_fn_2: CF,
            egraph: &EGraph<L, A>,
            id: Id,
        ) -> Option<(CF::Cost, RecExpr<L>)>
        where
            L: Language,
            A: Analysis<L>,
            CF: CostFunction<L>,
            CF::Cost: 'static + Ord,
        {
            use std::time::Instant;
            let t1 = Instant::now();
            let res1 = crate::sketch::extract::eclass_extract_sketch(s, cost_fn_1, egraph, id);
            let t2 = Instant::now();
            let res2 = crate::sketch::recursive::eclass_extract_sketch(s, cost_fn_2, egraph, id);
            let t3 = Instant::now();
            assert_eq!(res1.is_some(), res2.is_some());
            if let (Some((c1, _)), Some((c2, _))) = (&res1, &res2) {
                assert_eq!(c1, c2);
            };
            println!(
                "e-class analysis extraction took: {:?}",
                t2.duration_since(t1)
            );
            println!(
                "recursive descent extraction took: {:?}\n",
                t3.duration_since(t2)
            );
            res1
        }

        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let root_a = egraph.add_expr(&a_expr);
        let root_b = egraph.add_expr(&b_expr);

        egraph.rebuild();
        egraph.union(root_a, root_b);
        egraph.rebuild();

        let (best_cost, best_expr) =
            comp_eclass_extracted_sketch(&sketch, AstSize, AstSize, &egraph, root_a).unwrap();
        assert_eq!(best_cost, 4);
        assert_eq!(best_expr, a_expr);
    }
}
