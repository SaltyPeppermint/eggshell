use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;

use super::analysis::ExtractAnalysis;
use super::{analysis, utils};
use super::{Sketch, SketchNode};
use crate::sketch::hashcons::ExprHashCons;

/// More recursive, immutable version of `super::mutable::eclass_extract`
/// Also surprisingly faster
/// # Panics
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
pub fn eclass_extract<L, N, CF>(
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

#[expect(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::too_many_lines
)]
fn map_extract_rec<L, N, CF>(
    egraph_id: Id,
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
    if let Some(value) = memo.get(&(egraph_id, sketch_id)) {
        return value.clone();
    };

    let result = match &sketch[sketch_id] {
        // If the sketch says any, all the nodes in the eclass fulfill the sketch and
        // are valide solutions.
        SketchNode::Any => extracted.get(&egraph_id).cloned(),
        // if we have a specific node in the sketch, we need to check if there is such
        // a node in the current eclass under investigation and then check the children.
        SketchNode::Node(node) => {
            // Get the eclass we are currently checking
            let eclass = &egraph[egraph_id];

            // let mut candidates = Vec::new();
            let mnode = &node.clone().map_children(|_| Id::from(0));
            // We have a sketch (node) for this eclass and now we try to find
            // all the nodes in the eclass that fullfill that sketch.
            // We do this by checking the sketch for the current node (via the cloned mnode)
            // and then recursively checking if the children of the matching node holds
            // for the "sub-sketches" that are the children of the sketch_node
            let candidates = utils::flat_map_matching_node(eclass, mnode, |matched| {
                let mut matches = Vec::new();
                // Check if for each child of the sketch_node (themselves sketches),
                // the matched nodes children also hold for these child-sketches
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
                // Only if for every child the sketch holds it makes sense to continue
                if matches.len() == matched.len() {
                    // If all children match the sketch, they can be costed.
                    let to_match: HashMap<_, _> =
                        matched.children().iter().zip(matches.iter()).collect();
                    // Returns the valid subtree and its cost
                    Some((
                        cost_fn.cost(matched, |c| to_match[&c].0.clone()),
                        exprs.add(
                            matched
                                .clone()
                                .map_children(|child_id| to_match[&child_id].1),
                        ),
                    ))
                } else {
                    // Not all children hold, so this node does not fulfill the sketch
                    // Return None
                    None
                }
            });

            // We want to return the cheapest of the candidates valid
            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        SketchNode::Contains(inner_sketch_id) => {
            // avoid cycles
            // If we have visited the contains once, we do not need to
            // visit it again as the cost in our setup only goes up
            memo.insert((egraph_id, sketch_id), None);

            // Get the current EClass
            let eclass = &egraph[egraph_id];
            // Check recursively if the inner sketch in the contains()
            // is fulfilled by the current nodes in the eclass.
            // This gives us a vector of maximum length eclass.nodes.len().
            // Some may match, some dont but we can't stop here because the children
            // of those who do or dont could match the inner sketch!
            let mut candidates = map_extract_rec(
                egraph_id,
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

            // As previously stated we need to recursively iterate over the children in this eclass,
            // since they could fulfill the inner_sketch, thereby making their parent_nodes
            // fullfill the sketch, regardless if these parents directly fulfill the sketch.
            for enode in &eclass.nodes {
                // For each enode in the eclass, check if the children fullfil the contains
                // sketch. Notice how we are passing the sketch_id, not the inner_sketch_id!
                let matching_children = enode
                    .children()
                    .iter()
                    .filter_map(|&child_id| {
                        map_extract_rec(
                            child_id, sketch, sketch_id, cost_fn, egraph, exprs, extracted, memo,
                        )
                        // Cost the successful children
                        .map(|x| (child_id, x))
                    })
                    .collect::<Vec<_>>();
                // Collecting all the children, regardless if they fulfill the sketch
                // with their cost.
                // Remember, only one child has to fulfill sketch
                let children_any = enode
                    .children()
                    .iter()
                    .map(|&child_id| (child_id, extracted[&egraph.find(child_id)].clone()))
                    .collect::<Vec<_>>();

                // Iterating over all the matching children.
                // If any of them do, we can safely add the other children to the solution
                // since the loop wont run at all if there are zero matching children.
                // Put another way, if the outer loop ever runs the child_id==matching_id
                // has to be true for at least one
                for (matching_child_id, matching) in &matching_children {
                    let to_selected = children_any
                        .iter()
                        .map(|(child_id, any)| {
                            let selected = if child_id == matching_child_id {
                                matching
                            } else {
                                any
                            };
                            (*child_id, selected)
                        })
                        .collect::<HashMap<_, _>>();

                    candidates.push((
                        cost_fn.cost(enode, |c| to_selected[&c].0.clone()),
                        exprs.add(enode.clone().map_children(|c| to_selected[&c].1)),
                    ));
                }
            }

            candidates.into_iter().min_by(|x, y| x.0.cmp(&y.0))
        }
        // Rather simple: We check if either the fst or snd of the or pair fulfills
        // the sketch and we take the cheaper one
        SketchNode::Or(inner_sketch_ids) => inner_sketch_ids
            .iter()
            .filter_map(|inner_sketch_id| {
                map_extract_rec(
                    egraph_id,
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

    // cache the result
    memo.insert((egraph_id, sketch_id), result.clone());
    result
}
