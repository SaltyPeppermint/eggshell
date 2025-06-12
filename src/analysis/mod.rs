pub mod commutative_semigroup;
pub mod semilattice;

use std::hash::Hash;

use egg::{Analysis, EClass, EGraph, Id, Language};
use hashbrown::HashSet;

/// A data structure to maintain a queue of unique elements.
///
/// Notably, insert/pop operations have O(1) expected amortized runtime complexity.
///
/// Thanks @Bastacyclop for the implementation!
#[derive(Clone, PartialEq, Eq, Debug)]
struct UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    set: HashSet<T>, // hashbrown::
    queue: std::collections::VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: HashSet::default(),
            queue: std::collections::VecDeque::default(),
        }
    }
}

impl<U> FromIterator<U> for UniqueQueue<U>
where
    U: Eq + Hash + Clone,
{
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        let mut queue = Self::default();
        for t in iter {
            queue.insert(t);
        }
        queue
    }
}

impl<T> UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        res.as_ref().map(|t| self.set.remove(t));
        res
    }

    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}

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
