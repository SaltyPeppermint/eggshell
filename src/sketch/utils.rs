use std::mem::Discriminant;

use egg::{Analysis, EClass, EGraph, Id, Language};
use rustc_hash::{FxHashMap, FxHashSet};

/// Apply a function for each node in a eclass
/// Copied from <https://github.com/egraphs-good/egg/blob/347326d2e8ebbacca12d8e1398b86eff6dcfb2c5/src/machine.rs#L37>
/// Workaround so we don't use the private method in egg and can use the vanilla egg version and not
/// <https://github.com/egraphs-good/egg/compare/main...Bastacyclop:egg:sges>
pub fn for_each_matching_node<L, D>(
    eclass: &EClass<L, D>,
    node: &L,
    mut f: impl FnMut(&L) -> Result<(), ()>,
) -> Result<(), ()>
where
    L: Language,
{
    if eclass.nodes.len() < 50 {
        eclass
            .nodes
            .iter()
            .filter(|n| node.matches(n))
            .try_for_each(f)
    } else {
        debug_assert!(node.all(|id| id == Id::from(0)));
        debug_assert!(eclass.nodes.windows(2).all(|w| w[0] < w[1]));
        let mut start = eclass.nodes.binary_search(node).unwrap_or_else(|i| i);
        // let discrim = std::mem::discriminant(node);
        while start > 0 {
            if eclass.nodes[start - 1].matches(node) {
                start -= 1;
            } else {
                break;
            }
        }
        let mut matching = eclass.nodes[start..]
            .iter()
            .take_while(|&n| n.matches(node))
            .filter(|n| node.matches(n));
        debug_assert_eq!(
            matching.clone().count(),
            eclass.nodes.iter().filter(|n| node.matches(n)).count(),
            "matching node {:?}\nstart={}\n{:?} != {:?}\nnodes: {:?}",
            node,
            start,
            matching.clone().collect::<FxHashSet<_>>(),
            eclass
                .nodes
                .iter()
                .filter(|n| node.matches(n))
                .collect::<FxHashSet<_>>(),
            eclass.nodes
        );
        matching.try_for_each(&mut f)
    }
}

/// Workaround since an identical data struct `classes_by_op` is private in the egraph struct
/// Using just the discriminant is ok since it is again checked in the `for_each_matching_node` function
pub fn new_classes_by_op<L, N>(egraph: &EGraph<L, N>) -> FxHashMap<Discriminant<L>, FxHashSet<Id>>
where
    L: Language,
    N: Analysis<L>,
{
    let mut classes_by_op: FxHashMap<Discriminant<L>, FxHashSet<Id>> = FxHashMap::default();
    for class in egraph.classes() {
        for node in &class.nodes {
            let key = std::mem::discriminant(node);
            if let Some(ids) = classes_by_op.get_mut(&key) {
                ids.insert(class.id);
            } else {
                classes_by_op.insert(key, FxHashSet::default());
            }
        }
    }
    classes_by_op
}

pub fn classes_matching_op<'a, L: Language>(
    enode: &'a L,
    classes_by_op: &'a FxHashMap<Discriminant<L>, FxHashSet<Id>>,
) -> Option<&'a FxHashSet<Id>> {
    let key = std::mem::discriminant(enode);
    classes_by_op.get(&key)
}