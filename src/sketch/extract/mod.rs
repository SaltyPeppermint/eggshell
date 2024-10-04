pub mod mutable;
pub mod recursive;

use std::mem::Discriminant;

use egg::{Analysis, EGraph, Id, Language};
use hashbrown::{HashMap, HashSet};

use crate::analysis::semilattice::{SatisfiesContainsAnalysis, SemiLatticeAnalysis};

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

                SatisfiesContainsAnalysis.one_shot_analysis(egraph, &mut data);

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
    fn simple_mutable_extract_cost() {
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
            mutable::eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        assert_eq!(best_cost, 4);
        assert_eq!(best_expr, a_expr);
    }

    #[test]
    fn simple_rec_extract_cost() {
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
            recursive::eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        assert_eq!(best_cost, 4);
        assert_eq!(best_expr, a_expr);
    }

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

        let (c1, e1) = mutable::eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();
        let (c2, e2) = recursive::eclass_extract(&sketch, AstSize, &egraph, root_a).unwrap();

        assert_eq!(c1, c2);
        assert_eq!(e1, e2);
    }
}
