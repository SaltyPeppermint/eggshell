use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};

use crate::HashMap;

use super::analysis::ExtractAnalysis;
use super::hashcons::ExprHashCons;
use super::{analysis, utils};
use super::{Sketch, SketchNode};

pub fn eclass_extract_sketch<L, N, CF>(
    sketch: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, N>,
    id: Id,
) -> Option<(CF::Cost, RecExpr<L>)>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    assert!(egraph.clean);
    let mut memo = HashMap::<(Id, Id), Option<(CF::Cost, Id)>>::default();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = HashMap::default();
    let mut analysis = ExtractAnalysis::new(&mut exprs, &mut cost_fn);
    analysis::one_shot_analysis(egraph, &mut analysis, &mut extracted);

    let best_option = extract_rec(
        id,
        sketch,
        sketch_root,
        &mut cost_fn,
        egraph,
        &mut exprs,
        &extracted,
        &mut memo,
    );

    best_option.map(|(best_cost, best_id)| (best_cost, exprs.extract(best_id)))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn extract_rec<L, N, CF>(
    id: Id,
    sketch: &Sketch<L>,
    sid: Id,
    cost_fn: &mut CF,
    egraph: &EGraph<L, N>,
    exprs: &mut ExprHashCons<L>,
    extracted: &HashMap<Id, (CF::Cost, Id)>,
    memo: &mut HashMap<(Id, Id), Option<(CF::Cost, Id)>>,
) -> Option<(CF::Cost, Id)>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    if let Some(value) = memo.get(&(id, sid)) {
        return value.clone();
    };

    let result = match &sketch[sid] {
        SketchNode::Any => extracted.get(&id).cloned(),
        SketchNode::Node(node) => {
            let eclass = &egraph[id];
            let mut candidates = Vec::new();

            let mnode = &node.clone().map_children(|_| Id::from(0));
            let _ = utils::for_each_matching_node(eclass, mnode, |matched| {
                let mut matches = Vec::new();
                for (inner_sid, child_id) in node.children().iter().zip(matched.children()) {
                    if let Some(m) = extract_rec(
                        *child_id, sketch, *inner_sid, cost_fn, egraph, exprs, extracted, memo,
                    ) {
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

            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Contains(inner_sid) => {
            memo.insert((id, sid), None); // avoid cycles

            let eclass = &egraph[id];
            let mut candidates = Vec::new();
            candidates.extend(extract_rec(
                id, sketch, *inner_sid, cost_fn, egraph, exprs, extracted, memo,
            ));

            for enode in &eclass.nodes {
                let children_matching: Vec<_> = enode
                    .children()
                    .iter()
                    .filter_map(|&child_id| {
                        extract_rec(
                            child_id, sketch, sid, cost_fn, egraph, exprs, extracted, memo,
                        )
                        .map(move |x| (child_id, x))
                    })
                    .collect();
                let children_any: Vec<_> = enode
                    .children()
                    .iter()
                    .map(|&child_id| (child_id, extracted[&egraph.find(child_id)].clone()))
                    .collect();

                for (matching_child, matching) in &children_matching {
                    let mut to_selected = HashMap::default();

                    for (child, any) in &children_any {
                        let selected = if child == matching_child {
                            matching
                        } else {
                            any
                        };
                        to_selected.insert(child, selected);
                    }

                    candidates.push((
                        cost_fn.cost(enode, |c| to_selected[&c].0.clone()),
                        exprs.add(enode.clone().map_children(|c| to_selected[&c].1)),
                    ));
                }
            }

            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Or(inner_sids) => inner_sids
            .iter()
            .filter_map(|inner_sid| {
                extract_rec(
                    id, sketch, *inner_sid, cost_fn, egraph, exprs, extracted, memo,
                )
            })
            .min_by(|x, y| x.0.cmp(&y.0)),
    };

    memo.insert((id, sid), result.clone());
    result
}
