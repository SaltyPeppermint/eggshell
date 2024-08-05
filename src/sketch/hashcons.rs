use egg::{Id, Language, RecExpr};
use rustc_hash::{FxHashMap, FxHashSet};

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
pub(crate) struct ExprHashCons<L> {
    rec_expr: RecExpr<L>,
    memo: FxHashMap<L, Id>,
}

impl<L: Language> ExprHashCons<L> {
    pub(crate) fn new() -> Self {
        ExprHashCons {
            rec_expr: RecExpr::default(),
            memo: FxHashMap::default(),
        }
    }

    pub(crate) fn add(&mut self, node: L) -> Id {
        if let Some(id) = self.memo.get(&node) {
            *id
        } else {
            self.rec_expr.add(node)
        }
    }

    pub(crate) fn extract(&self, id: Id) -> RecExpr<L> {
        let all = self.rec_expr.as_ref();

        let mut used = FxHashSet::default();
        used.insert(id);
        for (i, node) in all.iter().enumerate().rev() {
            if used.contains(&Id::from(i)) {
                for c in node.children() {
                    used.insert(*c);
                }
                // node.for_each(|c| {
                //     used.insert(c);
                // });
            }
        }

        let mut fresh = RecExpr::default();
        let mut map = FxHashMap::<Id, Id>::default();
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
