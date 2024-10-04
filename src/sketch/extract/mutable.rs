use std::mem::Discriminant;

use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

use super::utils;
use super::{Sketch, SketchNode};
use crate::analysis::{ExtractAnalysis, ExtractContainsAnalysis, SemiLatticeAnalysis};
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

                analysis.one_shot_analysis(egraph, &mut data);

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
                let mut matching_ids = HashSet::new();
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
    analysis.one_shot_analysis(egraph, &mut extracted);

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
