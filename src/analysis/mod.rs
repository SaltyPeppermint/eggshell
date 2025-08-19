pub mod commutative_semigroup;
pub mod semilattice;

use egg::{Analysis, EClass, EGraph, Id, Language};

/// Return an iterator over a pairs of the parents node and their canonical `Id`
pub(crate) fn old_parents_iter<'a, L, N>(
    eclass: &'a EClass<L, N::Data>,
    egraph: &'a EGraph<L, N>,
) -> impl Iterator<Item = (&'a L, Id)>
where
    L: Language,
    N: Analysis<L>,
{
    let eclass_id = egraph.find(eclass.id);
    eclass.parents().flat_map(move |id| {
        egraph[id]
            .nodes
            .iter()
            .filter(move |n| {
                n.children()
                    .iter()
                    .any(|c_id| egraph.find(*c_id) == eclass_id)
            })
            .map(move |n| (n, egraph.find(id)))
    })
}
