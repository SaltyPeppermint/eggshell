use egg::{Id, Language, RecExpr};

#[derive(Debug, Clone)]
pub struct RecNode<L: Language> {
    pub node: L,
    pub children: Vec<RecNode<L>>,
}

impl<L: Language> RecNode<L> {
    pub fn new(node: L, children: Vec<RecNode<L>>) -> Self {
        Self { node, children }
    }
}

impl<L: Language> From<&RecExpr<L>> for RecNode<L> {
    fn from(value: &RecExpr<L>) -> Self {
        fn rec<LL: Language>(id: Id, rec_expr: &RecExpr<LL>) -> RecNode<LL> {
            let node = &rec_expr[id];
            RecNode {
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
