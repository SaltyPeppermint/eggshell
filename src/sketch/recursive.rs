use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};

use crate::HashMap;

use super::analysis::ExtractAnalysis;
use super::hashcons::ExprHashCons;
use super::{analysis, utils};
use super::{Sketch, SketchNode};

/// Recursive, mutable version of `super::extract::eclass_extract`
/// # Panics
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
pub fn for_each_eclass_extract<L, N, CF>(
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

#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::too_many_lines
)]
fn extract_rec<L, N, CF>(
    id: Id,
    sketch: &Sketch<L>,
    sketch_id: Id,
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
    if let Some(value) = memo.get(&(id, sketch_id)) {
        return value.clone();
    };

    let result = match &sketch[sketch_id] {
        SketchNode::Any => extracted.get(&id).cloned(),
        SketchNode::Node(node) => {
            let eclass = &egraph[id];

            let mut candidates = Vec::new();
            let mnode = &node.clone().map_children(|_| Id::from(0));
            let _ = utils::for_each_matching_node(eclass, mnode, |matched| {
                let mut matches = Vec::new();
                for (child_sketch_id, child_id) in node.children().iter().zip(matched.children()) {
                    if let Some(m) = extract_rec(
                        *child_id,
                        sketch,
                        *child_sketch_id,
                        cost_fn,
                        egraph,
                        exprs,
                        extracted,
                        memo,
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
                        exprs.add(
                            matched
                                .clone()
                                .map_children(|child_id| to_match[&child_id].1),
                        ),
                    ));
                }
                Ok(())
            });
            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Contains(inner_sketch_id) => {
            memo.insert((id, sketch_id), None); // avoid cycles

            let eclass = &egraph[id];
            let mut candidates = Vec::new();
            candidates.extend(extract_rec(
                id,
                sketch,
                *inner_sketch_id,
                cost_fn,
                egraph,
                exprs,
                extracted,
                memo,
            ));

            for enode in &eclass.nodes {
                let children_matching = enode
                    .children()
                    .iter()
                    .filter_map(|&child_id| {
                        extract_rec(
                            child_id, sketch, sketch_id, cost_fn, egraph, exprs, extracted, memo,
                        )
                        .map(move |x| (child_id, x))
                    })
                    .collect::<Vec<_>>();
                let children_any = enode
                    .children()
                    .iter()
                    .map(|&child_id| (child_id, extracted[&egraph.find(child_id)].clone()))
                    .collect::<Vec<_>>();

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
        SketchNode::Or(inner_sketch_ids) => inner_sketch_ids
            .iter()
            .filter_map(|inner_sketch_id| {
                extract_rec(
                    id,
                    sketch,
                    *inner_sketch_id,
                    cost_fn,
                    egraph,
                    exprs,
                    extracted,
                    memo,
                )
            })
            .min_by(|x, y| x.0.cmp(&y.0)),
    };

    memo.insert((id, sketch_id), result.clone());
    result
}

/// Recursive, immutable version of `super::extract::eclass_extract`
/// # Panics
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
pub fn map_eclass_extract<L, N, CF>(
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

    let best_option = map_extract_rec(
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

#[allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::too_many_lines
)]
fn map_extract_rec<L, N, CF>(
    id: Id,
    sketch: &Sketch<L>,
    sketch_id: Id,
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
    if let Some(value) = memo.get(&(id, sketch_id)) {
        return value.clone();
    };

    let result = match &sketch[sketch_id] {
        SketchNode::Any => extracted.get(&id).cloned(),
        SketchNode::Node(node) => {
            let eclass = &egraph[id];

            // let mut candidates = Vec::new();
            let mnode = &node.clone().map_children(|_| Id::from(0));
            let candidates = utils::flat_map_matching_node(eclass, mnode, |matched| {
                let mut matches = Vec::new();
                for (child_sketch_id, child_id) in node.children().iter().zip(matched.children()) {
                    if let Some(m) = map_extract_rec(
                        *child_id,
                        sketch,
                        *child_sketch_id,
                        cost_fn,
                        egraph,
                        exprs,
                        extracted,
                        memo,
                    ) {
                        matches.push(m);
                    } else {
                        break;
                    }
                }

                if matches.len() == matched.len() {
                    let to_match: HashMap<_, _> =
                        matched.children().iter().zip(matches.iter()).collect();
                    Some((
                        cost_fn.cost(matched, |c| to_match[&c].0.clone()),
                        exprs.add(
                            matched
                                .clone()
                                .map_children(|child_id| to_match[&child_id].1),
                        ),
                    ))
                } else {
                    None
                }
            });

            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Contains(inner_sketch_id) => {
            memo.insert((id, sketch_id), None); // avoid cycles

            let eclass = &egraph[id];
            let mut candidates = map_extract_rec(
                id,
                sketch,
                *inner_sketch_id,
                cost_fn,
                egraph,
                exprs,
                extracted,
                memo,
            )
            .into_iter()
            .collect::<Vec<_>>();

            for enode in &eclass.nodes {
                let children_matching = enode
                    .children()
                    .iter()
                    .filter_map(|&child_id| {
                        map_extract_rec(
                            child_id, sketch, sketch_id, cost_fn, egraph, exprs, extracted, memo,
                        )
                        .map(move |x| (child_id, x))
                    })
                    .collect::<Vec<_>>();
                let children_any = enode
                    .children()
                    .iter()
                    .map(|&child_id| (child_id, extracted[&egraph.find(child_id)].clone()))
                    .collect::<Vec<_>>();

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
        SketchNode::Or(inner_sketch_ids) => inner_sketch_ids
            .iter()
            .filter_map(|inner_sketch_id| {
                map_extract_rec(
                    id,
                    sketch,
                    *inner_sketch_id,
                    cost_fn,
                    egraph,
                    exprs,
                    extracted,
                    memo,
                )
            })
            .min_by(|x, y| x.0.cmp(&y.0)),
    };

    memo.insert((id, sketch_id), result.clone());
    result
}

#[cfg(test)]
mod tests {
    use egg::{AstSize, EGraph, RecExpr, SymbolLang};

    use super::*;
    use crate::sketch::{
        extract::{self, eclass_extract},
        Sketch,
    };

    #[test]
    fn analysis_vs_recursive() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let root_a = egraph.add_expr(&a_expr);
        let root_b = egraph.add_expr(&b_expr);

        egraph.rebuild();
        egraph.union(root_a, root_b);
        egraph.rebuild();

        let (c1, e1) = extract::eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        let (c2, e2) = eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();

        assert_eq!(c1, c2);
        assert_eq!(e1, e2);
    }

    #[test]
    fn map_vs_for_each() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let root_a = egraph.add_expr(&a_expr);
        let root_b = egraph.add_expr(&b_expr);

        egraph.rebuild();
        egraph.union(root_a, root_b);
        egraph.rebuild();

        let (c1, e1) = eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        let (c2, e2) = map_eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();

        assert_eq!(c1, c2);
        assert_eq!(e1, e2);
    }
}
