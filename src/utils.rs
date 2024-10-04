use std::hash::Hash;

use egg::{Id, Language, RecExpr};
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

pub(crate) trait Tree: Sized {
    fn children(&self) -> &[Self];

    fn children_mut(&mut self) -> &mut [Self];

    fn size(&self) -> usize {
        1 + self.children().iter().map(|c| c.size()).sum::<usize>()
    }

    fn depth(&self) -> usize {
        1 + self.children().iter().map(|c| c.depth()).max().unwrap_or(0)
    }
}

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
#[derive(Debug)]
pub(crate) struct ExprHashCons<L> {
    expr: RecExpr<L>,
    memo: HashMap<L, Id>,
}

impl<L: Language> ExprHashCons<L> {
    pub(crate) fn new() -> Self {
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
