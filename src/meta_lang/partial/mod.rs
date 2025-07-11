mod error;
mod parse;

use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::mem::Discriminant;

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use crate::node::RecNode;
use crate::rewrite_system::{LangExtras, SymbolInfo, SymbolType};

pub use error::PartialError;
pub use parse::{count_expected_tokens, partial_parse};

/// Simple alias
pub type PartialRecExpr<L> = RecExpr<PartialLang<L>>;

#[derive(
    Debug,
    PartialEq,
    Clone,
    Hash,
    Ord,
    Eq,
    PartialOrd,
    Serialize,
    Deserialize,
    EnumDiscriminants,
    EnumCount,
)]
#[strum_discriminants(derive(EnumIter))]
pub enum PartialLang<L: Language> {
    /// Finshed parts represented by [`SketchNode`]
    Pad,
    Finished(L),
}

impl<L> PartialLang<L>
where
    L: Language + LangExtras + FromOp,
    L::Error: Display,
{
    pub fn lower(higher: &RecExpr<Self>) -> Result<RecExpr<L>, error::PartialError<L>> {
        higher
            .into_iter()
            .map(|partial_node| match partial_node {
                PartialLang::Finished(inner) => Ok(inner.to_owned()),
                PartialLang::Pad => Err(error::PartialError::NoLowering(partial_node.to_string())),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.into())
    }
}

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = std::mem::discriminant(self);
        match self {
            PartialLang::Finished(inner) => (discr, Some(inner.discriminant())),
            Self::Pad => (discr, None),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }

    fn children(&self) -> &[Id] {
        match self {
            PartialLang::Finished(inner) => inner.children(),
            Self::Pad => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            PartialLang::Finished(inner) => inner.children_mut(),
            Self::Pad => &mut [],
        }
    }
}

impl<L: Language + LangExtras> LangExtras for PartialLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        if let PartialLang::Finished(finished) = self {
            finished.symbol_info()
        } else {
            let position = PartialLangDiscriminants::iter()
                .position(|x| x == self.into())
                .unwrap();
            SymbolInfo::new(position + L::NUM_SYMBOLS, SymbolType::MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["<pad>"];
        s.extend(L::operators());
        s
    }

    fn pretty_string(&self) -> String {
        match self {
            PartialLang::Finished(inner) => inner.pretty_string(),
            PartialLang::Pad => self.to_string(),
        }
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PartialLang::Finished(inner) => write!(f, "{inner}"),
            Self::Pad => write!(f, "<pad>"),
        }
    }
}

impl<L: Language> RecNode<PartialLang<L>> {
    fn new_empty() -> Self {
        RecNode::new(PartialLang::Pad, Vec::new())
    }

    fn find_next_open(&mut self) -> Option<&mut Self> {
        let mut queue = VecDeque::new();
        queue.push_back(self);
        while let Some(x) = queue.pop_front() {
            if matches!(x.node, PartialLang::Pad) {
                return Some(x);
            }
            for c in &mut x.children {
                queue.push_back(c);
            }
        }
        None
    }
}

impl<L: Language> From<RecNode<PartialLang<L>>> for RecExpr<PartialLang<L>> {
    fn from(root: RecNode<PartialLang<L>>) -> Self {
        fn rec<LL: Language>(
            mut curr: RecNode<PartialLang<LL>>,
            vec: &mut Vec<PartialLang<LL>>,
        ) -> Id {
            let c = curr.children.into_iter().map(|c| rec(c, vec));
            for (dummy_id, c_id) in curr.node.children_mut().iter_mut().zip(c) {
                *dummy_id = c_id;
            }
            let id = Id::from(vec.len());
            vec.push(curr.node);
            id
        }

        let mut stack = Vec::new();
        rec(root, &mut stack);
        RecExpr::from(stack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_lang::SketchLang;
    use crate::rewrite_system::LangExtras;
    use crate::rewrite_system::halide::HalideLang;

    #[test]
    fn operators() {
        let known_operators = vec![
            "<pad>", "?", "contains", "or", "+", "-", "*", "/", "%", "max", "min", "<", ">", "!",
            "<=", ">=", "==", "!=", "||", "&&",
        ];
        assert_eq!(
            PartialLang::<SketchLang<HalideLang>>::operators(),
            known_operators
        );
    }
}
