use egg::{Id, Language, RecExpr};

#[derive(Debug, Clone)]
pub struct OwnedRecNode<L: Language, T> {
    pub node: L,
    pub children: Vec<OwnedRecNode<L, T>>,
    pub additional_data: T,
}

impl<L: Language, T> OwnedRecNode<L, T> {
    pub fn new(node: L, children: Vec<OwnedRecNode<L, T>>, additional_data: T) -> Self {
        Self {
            node,
            children,
            additional_data,
        }
    }
}

impl<L: Language, T: Clone> From<(&RecExpr<L>, &[T])> for OwnedRecNode<L, T> {
    fn from(value: (&RecExpr<L>, &[T])) -> Self {
        fn rec<LL: Language, TT: Clone>(
            id: Id,
            rec_expr: &RecExpr<LL>,
            additional_data: &[TT],
        ) -> OwnedRecNode<LL, TT> {
            let node = &rec_expr[id];
            OwnedRecNode {
                node: node.to_owned(),
                children: node
                    .children()
                    .iter()
                    .map(|c_id| rec(*c_id, rec_expr, additional_data))
                    .collect(),
                additional_data: additional_data[usize::from(id)].to_owned(),
            }
        }
        rec(value.0.root(), value.0, value.1)
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
