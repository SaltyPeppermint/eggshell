use egg::{Analysis, EGraph, Guide, Id, Language, RecExpr};

#[derive(Debug)]
pub struct ConcreteGuide<L: Language>(RecExpr<L>);

impl<L: Language> ConcreteGuide<L> {
    pub fn new(rec_expr: RecExpr<L>) -> Self {
        Self(rec_expr)
    }
}

impl<L: Language, N: Analysis<L>> Guide<L, N> for ConcreteGuide<L> {
    fn check(&self, egraph: &EGraph<L, N>, id: Id) -> Option<RecExpr<L>> {
        if egraph.lookup_expr(&self.0)? == id {
            Some(self.0.clone())
        } else {
            None
        }
    }
}
