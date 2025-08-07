use egg::{Analysis, AstSize, EGraph, Guide, Id, Language, RecExpr};

use crate::meta_lang::sketch::{eclass_extract, satisfies_sketch};

use super::Sketch;

#[derive(Debug)]
pub struct SketchGuide<L: Language>(Sketch<L>);

impl<L: Language> SketchGuide<L> {
    pub fn new(rec_expr: Sketch<L>) -> Self {
        Self(rec_expr)
    }
}

impl<L: Language, N: Analysis<L>> Guide<L, N> for SketchGuide<L> {
    fn check(&self, egraph: &EGraph<L, N>, id: Id) -> Option<RecExpr<L>> {
        let eclass_id = *satisfies_sketch(&self.0, egraph).get(&id)?;
        let (_, rec_expr) = eclass_extract(&self.0, AstSize, egraph, eclass_id)?;
        Some(rec_expr)
    }
}
