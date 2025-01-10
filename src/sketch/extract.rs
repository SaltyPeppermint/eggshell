use std::fmt::Debug;

use egg::{Analysis, CostFunction, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

use super::{Sketch, SketchNode};
use crate::analysis::semilattice::{
    ExtractAnalysis, SatisfiesContainsAnalysis, SemiLatticeAnalysis,
};
use crate::utils::ExprHashCons;

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
        memo: &mut HashMap<Id, HashSet<Id>>,
    ) -> HashSet<Id> {
        if let Some(value) = memo.get(&sketch_id) {
            return value.clone();
        };

        let result = match &sketch[sketch_id] {
            // All nodes in the egraph fulfill the Any sketch
            SketchNode::Any => egraph
                .classes()
                .map(|eclass| {
                    // No egraph.find since we are taking the id directly from the eclass
                    eclass.id
                })
                .collect(),
            SketchNode::Node(sketch_node) => {
                // Get all nodes fullfilling the children of the current sketch_node
                // (Themselves sketches)
                // No egraph.find since we are only ever returning canonical ids
                let children_matches = sketch_node
                    .children()
                    .iter()
                    .map(|child_sketch_id| rec(sketch, *child_sketch_id, egraph, memo))
                    .collect::<Vec<_>>();

                // Get all eclasses that contain the note required by the sketch and iterate over these classes.
                // No egraph.find since classes_by_op only contain hashsets of canonical ids
                if let Some(potential_ids) = egraph.classes_for_op(&sketch_node.discriminant()) {
                    potential_ids
                        .filter(|&id| {
                            // Get the eclass corresponding to the id
                            let eclass = &egraph[id];
                            // Iterate over the nodes
                            eclass
                                .nodes
                                .iter()
                                // We are only interested in the nodes that is the ones matching the sketch
                                .filter(|n| sketch_node.matches(n))
                                // We check if all the children of these nodes also are in the set of nodes fulfilling
                                // the child sketches.
                                .all(|matched| {
                                    // Again, no find necessary since the class_by_op does that for us
                                    matched.children().iter().zip(children_matches.iter()).all(
                                        |(child_id, child_matches)| {
                                            child_matches.contains(child_id)
                                        },
                                    )
                                })
                        })
                        .collect()
                } else {
                    HashSet::new()
                }
            }
            SketchNode::Contains(inner_sketch_id) => {
                let contained_matched = rec(sketch, *inner_sketch_id, egraph, memo);

                // No egraph.find since we are only ever returning canonical ids
                let mut data = egraph
                    .classes()
                    .map(|eclass| {
                        // No egraph.find since we are taking the id directly from the eclass
                        (eclass.id, contained_matched.contains(&eclass.id))
                    })
                    .collect::<HashMap<_, bool>>();

                SatisfiesContainsAnalysis.one_shot_analysis(egraph, &mut data);

                data.iter()
                    .filter_map(|(&id, &is_match)| if is_match { Some(id) } else { None })
                    .collect()
            }
            SketchNode::Or(inner_sketch_ids) => {
                // No egraph.find since we are only ever returning canonical ids
                let matches = inner_sketch_ids
                    .iter()
                    .map(|inner_sketch_id| rec(sketch, *inner_sketch_id, egraph, memo));
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
    rec(sketch, sketch_root, egraph, &mut memo)
}

/// More recursive, immutable version of `super::mutable::eclass_extract`.
/// Also surprisingly faster.
///
/// # Panics
///
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
#[expect(clippy::too_many_lines)]
pub fn eclass_extract<L, N, CF>(
    sketch: &Sketch<L>,
    mut cost_fn: CF,
    egraph: &EGraph<L, N>,
    id: Id,
) -> Option<(CF::Cost, RecExpr<L>)>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L> + Debug,
    CF::Cost: Ord,
{
    /// Recursion for `eclass_extract`
    #[expect(
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::too_many_lines
    )]
    fn rec<L, N, CF>(
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
            SketchNode::Node(sketch_node) => {
                // Get the eclass from the egraph we are currently checking
                let eclass = &egraph[egraph_id];

                // We have a sketch (node) for this eclass and now we try to find
                // all the nodes in the eclass that fullfill that sketch.
                // We do this by checking the sketch for the current node
                // and then recursively checking if the children of the matching node holds
                // for the "sub-sketches" that are the children of the sketch_node
                eclass
                    .nodes
                    .iter()
                    .filter(|n| sketch_node.matches(n))
                    .filter_map(|matched| {
                        // Check for each child if it matches the children
                        // of the sketch_node (themselves sketches),
                        let to_matches = matched
                            .children()
                            .iter()
                            // We need this since children of matched can be non-canonical
                            .map(|child_id| egraph.find(*child_id))
                            // Sketches are only recexpr and therefore only have canonical ids
                            .zip(sketch_node.children())
                            .map(|(child_id, child_sketch_id)| {
                                Some((
                                    child_id,
                                    rec(
                                        child_id,
                                        sketch,
                                        *child_sketch_id,
                                        cost_fn,
                                        egraph,
                                        exprs,
                                        extracted,
                                        memo,
                                    )?,
                                ))
                            })
                            // Only if for every child the sketch holds it makes sense to continue
                            // Note the ? to return a None if not all children match
                            // If all goes well we end up with a hashmap of each child_id of a node and its result
                            .collect::<Option<HashMap<_, _>>>()?;

                        // Returns the valid subtree and its cost
                        // We again need the find here since the child_ids can be non-canonical
                        Some((
                            cost_fn.cost(matched, |c| to_matches[&egraph.find(c)].0.clone()),
                            exprs.add(
                                matched
                                    .clone()
                                    .map_children(|child_id| to_matches[&egraph.find(child_id)].1),
                            ),
                        ))
                    })
                    // We want to return the cheapest of the candidates valid
                    .min_by(|x, y| x.0.cmp(&y.0))
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
                let mut candidates = rec(
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
                        // Needed since children can be non-canonical
                        .map(|child_id| egraph.find(*child_id))
                        .filter_map(|child_id| {
                            rec(
                                child_id, sketch, sketch_id, cost_fn, egraph, exprs, extracted,
                                memo,
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
                        // Needed since children can be non-canonical
                        .map(|child_id| egraph.find(*child_id))
                        .map(|child_id| {
                            // We need the find since child_ids can be non-canonical
                            (child_id, extracted[&egraph.find(child_id)].clone())
                        })
                        .collect::<Vec<_>>();

                    // Iterating over all the matching children.
                    // If any of them do, we can safely add the other children to the solution
                    // since the loop wont run at all if there are zero matching children.
                    // Put another way, if the outer loop ever runs the child_id==matching_id
                    // has to be true for at least one
                    // Also, all entries in matching_children and children_any are canonicalized so no worry here
                    for (matching_child_id, matching) in &matching_children {
                        let to_selected = children_any
                            .iter()
                            .map(|(child_id, any)| {
                                // True at least once otherwise the outer loop would have never run
                                if child_id == matching_child_id {
                                    (*child_id, matching)
                                } else {
                                    (*child_id, any)
                                }
                            })
                            .collect::<HashMap<_, _>>();

                        candidates.push((
                            // We again need the find here since the child_ids can be non-canonical
                            cost_fn.cost(enode, |c| to_selected[&egraph.find(c)].0.clone()),
                            exprs.add(
                                enode
                                    .clone()
                                    .map_children(|c| to_selected[&egraph.find(c)].1),
                            ),
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
                    rec(
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

    assert!(egraph.clean);
    let mut memo = HashMap::<(Id, Id), Option<(CF::Cost, Id)>>::default();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    let mut exprs = ExprHashCons::new();

    let mut extracted = HashMap::default();
    let mut analysis = ExtractAnalysis::new(&mut exprs, &mut cost_fn);
    analysis.one_shot_analysis(egraph, &mut extracted);

    let best_option = rec(
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

#[cfg(test)]
mod tests {
    use egg::{AstSize, RecExpr, SymbolLang};

    use super::*;

    #[test]
    fn simple_contains() {
        let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();

        let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
        let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();
        let c_expr = "(j (f x))".parse::<RecExpr<SymbolLang>>().unwrap();
        let d_expr = "(h (h x))".parse::<RecExpr<SymbolLang>>().unwrap();

        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add_expr(&a_expr);
        let b = egraph.add_expr(&b_expr);
        let c = egraph.add_expr(&c_expr);
        let d = egraph.add_expr(&d_expr);

        egraph.rebuild();

        let sat = satisfies_sketch(&sketch, &egraph);

        assert!(sat.contains(&a));
        assert!(sat.contains(&b));
        assert!(sat.contains(&c));
        assert!(!sat.contains(&d));
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
