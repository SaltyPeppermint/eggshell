use egg::{Id, Language, RecExpr};

#[derive(Debug, Clone)]
pub struct OwnedRecNode<L: Language> {
    pub node: L,
    pub children: Vec<OwnedRecNode<L>>,
}

impl<L: Language> OwnedRecNode<L> {
    pub fn new(node: L, children: Vec<OwnedRecNode<L>>) -> Self {
        Self { node, children }
    }
}

impl<L: Language> From<&RecExpr<L>> for OwnedRecNode<L> {
    fn from(value: &RecExpr<L>) -> Self {
        fn rec<LL: Language>(id: Id, rec_expr: &RecExpr<LL>) -> OwnedRecNode<LL> {
            let node = &rec_expr[id];
            OwnedRecNode {
                node: node.to_owned(),
                children: node
                    .children()
                    .iter()
                    .map(|c_id| rec(*c_id, rec_expr))
                    .collect(),
            }
        }
        rec(value.root(), value)
    }
}

pub struct BorrowingRecNode<'a, L: Language> {
    node: &'a L,
    index: usize,
    children: Vec<BorrowingRecNode<'a, L>>,
}

impl<L: Language> BorrowingRecNode<'_, L> {
    pub fn leftmost(&self) -> &Self {
        self.children.first().unwrap_or(self)
    }

    pub fn children(&self) -> &[BorrowingRecNode<L>] {
        &self.children
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn node(&self) -> &L {
        self.node
    }
}

impl<'a, L: Language> From<&'a RecExpr<L>> for BorrowingRecNode<'a, L> {
    fn from(value: &'a RecExpr<L>) -> Self {
        fn rec<'aa, LL: Language>(
            rec_expr: &'aa RecExpr<LL>,
            id: Id,
            index: &mut usize,
        ) -> BorrowingRecNode<'aa, LL> {
            let curr_index = *index;
            *index += 1;
            let children = rec_expr[id]
                .children()
                .iter()
                .map(|c_id| rec(rec_expr, *c_id, index))
                .collect();

            BorrowingRecNode {
                node: &rec_expr[id],
                children,
                index: curr_index,
            }
        }
        let mut index = 0;
        rec(value, value.root(), &mut index)
    }
}
