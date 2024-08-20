use std::{fmt::Debug, hash::Hash};

use egg::{AstDepth, AstSize, CostFunction, Language};

use crate::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct AstSize2;

impl<L: Language> CostFunction<L> for AstSize2 {
    type Cost = usize;

    #[inline]
    fn cost<C>(&mut self, enode: &L, costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let mut inner = AstSize;
        inner.cost(enode, costs)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AstDepth2;

impl<L: Language> CostFunction<L> for AstDepth2 {
    type Cost = usize;

    #[inline]
    fn cost<C>(&mut self, enode: &L, costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let mut inner = AstDepth;
        inner.cost(enode, costs)
    }
}

/** A data structure to maintain a queue of unique elements.

Notably, insert/pop operations have O(1) expected amortized runtime complexity.

Thanks @Bastacyclop for the implementation!
*/
#[derive(Clone, PartialEq, Eq)]
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

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }
}
