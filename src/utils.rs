use std::collections::VecDeque;
use std::hash::Hash;

use egg::{Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct ExprHashCons<L: Hash + Eq + Language> {
    node_store: Vec<L>,
    memo: HashMap<L, usize>,
}

impl<L: Language> ExprHashCons<L> {
    pub fn new() -> Self {
        ExprHashCons {
            node_store: Vec::new(),
            memo: HashMap::default(),
        }
    }

    pub(crate) fn add(&mut self, node: L) -> usize {
        if let Some(id) = self.memo.get(&node) {
            return *id;
        }
        let new_id = self.node_store.len();
        self.node_store.push(node.clone());
        self.memo.insert(node, new_id);
        new_id
    }

    pub(crate) fn extract(&self, id: usize) -> RecExpr<L> {
        let mut used = HashSet::new();
        used.insert(id);
        for (i, node) in self.node_store.iter().enumerate().rev() {
            if used.contains(&i) {
                used.extend(node.children().iter().map(|c_id| usize::from(*c_id)));
            }
        }

        let mut fresh = RecExpr::default();
        let mut map = HashMap::<Id, Id>::default();
        for (i, node) in self.node_store.iter().enumerate() {
            if used.contains(&i) {
                let fresh_node = node.clone().map_children(|c| map[&c]);
                let fresh_id = fresh.add(fresh_node);
                map.insert(Id::from(i), fresh_id);
            }
        }

        fresh
    }
}

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
    queue: VecDeque<T>,
}

impl<T> Default for UniqueQueue<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        UniqueQueue {
            set: HashSet::default(),
            queue: VecDeque::default(),
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
        if let Some(t) = &res {
            self.set.remove(t);
        }
        res
    }

    #[expect(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}
