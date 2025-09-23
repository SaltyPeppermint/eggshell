use egg::{Analysis, EGraph, Id, Language};
use hashbrown::{HashMap, HashSet};

use super::{Sketch, SketchLang};
use crate::analysis::semilattice::{
    SatisfiesContainsAnalysis, SatisfiesOnlyContainsAnalysis, SemiLatticeAnalysis,
};

/// Is the `id` e-class of `egraph` representing at least one program satisfying `s`?
pub fn eclass_contains<L: Language, A: Analysis<L>>(
    sketch: &Sketch<L>,
    egraph: &EGraph<L, A>,
    id: Id,
) -> bool {
    contains(sketch, egraph).contains(&id)
}

/// Returns the set of e-classes of `egraph` that represent at least one program satisfying `s`.
/// # Panics
/// Panics if the egraph isn't clean.
/// Only give it clean egraphs!
pub fn contains<L: Language, A: Analysis<L>>(
    sketch: &Sketch<L>,
    egraph: &EGraph<L, A>,
) -> HashSet<Id> {
    assert!(egraph.clean);
    let mut memo = HashMap::<Id, HashSet<Id>>::default();
    // let sketch_nodes = s.as_ref();
    let sketch_root = Id::from(sketch.as_ref().len() - 1);
    rec_contains(sketch, sketch_root, egraph, &mut memo)
}

fn rec_contains<L: Language, A: Analysis<L>>(
    s_nodes: &[SketchLang<L>],
    s_index: Id,
    egraph: &EGraph<L, A>,
    memo: &mut HashMap<Id, HashSet<Id>>,
) -> HashSet<Id> {
    if let Some(value) = memo.get(&s_index) {
        return value.clone();
    };

    let result = match &s_nodes[usize::from(s_index)] {
        SketchLang::Any => egraph.classes().map(|c| c.id).collect(),
        SketchLang::Node(node) => {
            let children_matches = node
                .children()
                .iter()
                .map(|sid| rec_contains(s_nodes, *sid, egraph, memo))
                .collect::<Vec<_>>();

            if let Some(potential_ids) = egraph.classes_for_op(&node.discriminant()) {
                potential_ids
                    .filter(|&id| {
                        let eclass = &egraph[id];

                        let mnode = &node.clone().map_children(|_| Id::from(0));
                        eclass
                            .for_each_matching_node(mnode, |matched| {
                                let children_match = children_matches
                                    .iter()
                                    .zip(matched.children())
                                    .all(|(matches, id)| matches.contains(id));
                                if children_match { Err(()) } else { Ok(()) }
                            })
                            .is_err()
                    })
                    .collect()
            } else {
                HashSet::default()
            }
        }
        SketchLang::Contains(sid) => {
            let contained_matched = rec_contains(s_nodes, *sid, egraph, memo);

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matched.contains(&eclass.id)))
                .collect::<HashMap<_, _>>();

            SatisfiesContainsAnalysis.one_shot_analysis(egraph, &mut data);

            data.iter()
                .flat_map(|(&id, &is_match)| if is_match { Some(id) } else { None })
                .collect()
        }
        SketchLang::OnlyContains(sid) => {
            let contained_matched = rec_contains(s_nodes, *sid, egraph, memo);

            let mut data = egraph
                .classes()
                .map(|eclass| (eclass.id, contained_matched.contains(&eclass.id)))
                .collect::<HashMap<_, _>>();

            SatisfiesOnlyContainsAnalysis.one_shot_analysis(egraph, &mut data);

            data.iter()
                .flat_map(|(&id, &is_match)| if is_match { Some(id) } else { None })
                .collect()
        }
        SketchLang::Or(sids) => {
            let matches = sids
                .iter()
                .map(|sid| rec_contains(s_nodes, *sid, egraph, memo));
            matches
                .reduce(|a, b| a.union(&b).cloned().collect())
                .expect("empty or sketch")
        }
    };

    memo.insert(s_index, result.clone());
    result
}

#[cfg(test)]
mod tests {
    use egg::{RecExpr, SymbolLang};

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

        let sat = contains(&sketch, &egraph);

        assert!(sat.contains(&a));
        assert!(sat.contains(&b));
        assert!(sat.contains(&c));
        assert!(!sat.contains(&d));
    }
}
