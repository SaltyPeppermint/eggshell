use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use rustc_hash::FxHashMap;

use super::analysis::ExtractAnalysis;
use super::hashcons::ExprHashCons;
use super::{analysis, utils};
use super::{Sketch, SketchNode};

pub fn eclass_extract_sketch<L, N, CF>(
    s: &Sketch<L>,
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
    let mut memo = FxHashMap::<(Id, Id), Option<(CF::Cost, Id)>>::default();
    let sketch_nodes = s.as_ref();
    let sketch_root = Id::from(sketch_nodes.len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = FxHashMap::default();
    let mut analysis = ExtractAnalysis::new(&mut exprs, &mut cost_fn);
    analysis::one_shot_analysis(egraph, &mut analysis, &mut extracted);

    let best_option = extract_rec(
        id,
        sketch_nodes,
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
    s_nodes: &[SketchNode<L>],
    s_index: Id,
    cost_fn: &mut CF,
    egraph: &EGraph<L, N>,
    exprs: &mut ExprHashCons<L>,
    extracted: &FxHashMap<Id, (CF::Cost, Id)>,
    memo: &mut FxHashMap<(Id, Id), Option<(CF::Cost, Id)>>,
) -> Option<(CF::Cost, Id)>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Ord,
{
    if let Some(value) = memo.get(&(id, s_index)) {
        return value.clone();
    };

    let result = match &s_nodes[usize::from(s_index)] {
        SketchNode::Any => extracted.get(&id).cloned(),
        SketchNode::Node(node) => {
            let eclass = &egraph[id];
            let mut candidates = Vec::new();

            let mnode = &node.clone().map_children(|_| Id::from(0));
            let _ = utils::for_each_matching_node(eclass, mnode, |matched| {
                let mut matches = Vec::new();
                for (sid, id) in node.children().iter().zip(matched.children()) {
                    if let Some(m) =
                        extract_rec(*id, s_nodes, *sid, cost_fn, egraph, exprs, extracted, memo)
                    {
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

            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Contains(sid) => {
            memo.insert((id, s_index), None); // avoid cycles

            let eclass = &egraph[id];
            let mut candidates = Vec::new();
            candidates.extend(extract_rec(
                id, s_nodes, *sid, cost_fn, egraph, exprs, extracted, memo,
            ));

            for enode in &eclass.nodes {
                let children_matching: Vec<_> = enode
                    .children()
                    .iter()
                    .filter_map(|&c| {
                        extract_rec(c, s_nodes, s_index, cost_fn, egraph, exprs, extracted, memo)
                            .map(move |x| (c, x))
                    })
                    .collect();
                let children_any: Vec<_> = enode
                    .children()
                    .iter()
                    .map(|&c| (c, extracted[&egraph.find(c)].clone()))
                    .collect();

                for (matching_child, matching) in &children_matching {
                    let mut to_selected = FxHashMap::default();

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
        SketchNode::Or(sids) => sids
            .iter()
            .filter_map(|sid| {
                extract_rec(id, s_nodes, *sid, cost_fn, egraph, exprs, extracted, memo)
            })
            .min_by(|x, y| x.0.cmp(&y.0)),
    };

    memo.insert((id, s_index), result.clone());
    result
}