use std::hash::Hash;

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

/// A data structure to maintain a queue of unique elements.
///
/// Notably, insert/pop operations have O(1) expected amortized runtime complexity.
///
/// Thanks @Bastacyclop for the implementation!
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct UniqueQueue<T>
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

// // HELPER TO CUT OFF N LAST ELEMENTS OF A SLICE
// pub(crate) const fn cutoff_slice<T>(slice: &[T], cutoff: usize) -> &[T] {
//     let mut slice = slice;
//     let max_len = slice.len() - cutoff;
//     loop {
//         if slice.len() == max_len {
//             return slice;
//         }

//         slice = match slice {
//             [rest @ .., _last] => rest,
//             _ => panic!("Index out of bounds"),
//         }
//     }
// }

// pub(crate) trait Tree: Sized {
//     fn children(&self) -> &[Self];

//     fn size(&self) -> usize {
//         1 + self.children().iter().map(|c| c.size()).sum::<usize>()
//     }

//     fn depth(&self) -> usize {
//         1 + self.children().iter().map(|c| c.depth()).max().unwrap_or(0)
//     }
// }

// impl<L: Language> Tree for RecExpr<L> {
//     fn children(&self) -> &[Self] {
//         self.children()
//     }
// }

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ExprHashCons<L: Hash + Eq> {
    expr: RecExpr<L>,
    memo: HashMap<L, Id>,
}

impl<L: Language> ExprHashCons<L> {
    pub fn new() -> Self {
        ExprHashCons {
            expr: RecExpr::default(),
            memo: HashMap::default(),
        }
    }

    pub(crate) fn add(&mut self, node: L) -> Id {
        if let Some(id) = self.memo.get(&node) {
            *id
        } else {
            self.expr.add(node)
        }
    }

    pub(crate) fn extract(&self, id: Id) -> RecExpr<L> {
        let all = self.expr.as_ref();

        let mut used = HashSet::new();
        used.insert(id);
        for (i, node) in all.iter().enumerate().rev() {
            if used.contains(&Id::from(i)) {
                for c in node.children() {
                    used.insert(*c);
                }
            }
        }

        let mut fresh = RecExpr::default();
        let mut map = HashMap::<Id, Id>::default();
        for (i, node) in all.iter().enumerate() {
            if used.contains(&Id::from(i)) {
                let fresh_node = node.clone().map_children(|c| map[&c]);
                let fresh_id = fresh.add(fresh_node);
                map.insert(Id::from(i), fresh_id);
            }
        }

        fresh
    }
}

impl<L: Language> Default for ExprHashCons<L> {
    fn default() -> Self {
        Self::new()
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
