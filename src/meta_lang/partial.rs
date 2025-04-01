// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::MetaLangError;
use crate::trs::SymbolType::MetaSymbol;
use crate::trs::{MetaInfo, SymbolInfo};
// use crate::typing::{Type, Typeable, TypingInfo};

/// Simple alias
pub type PartialTerm<L> = RecExpr<PartialLang<L>>;

#[derive(
    Debug,
    Hash,
    PartialEq,
    Eq,
    Clone,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    EnumDiscriminants,
    EnumCount,
)]
#[strum_discriminants(derive(EnumIter))]
pub enum PartialLang<L: Language> {
    /// Finshed parts represented by [`SketchNode`]
    Finished(L),
    Placeholder,
}

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialLang::Finished(x) => (discr, Some(x.discriminant())),
            PartialLang::Placeholder => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Placeholder => &[],
            Self::Finished(n) => n.children(),
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Placeholder => &mut [],
            Self::Finished(n) => n.children_mut(),
        }
    }
}

impl<L: Language + MetaInfo> MetaInfo for PartialLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        if let PartialLang::Finished(l) = self {
            l.symbol_info()
        } else {
            let position = PartialLangDiscriminants::iter()
                .position(|x| x == self.into())
                .unwrap();
            SymbolInfo::new(position + L::NUM_SYMBOLS, MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["[placeholder]"];
        s.extend(L::operators());
        s
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placeholder => write!(f, "[placeholder]"),
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
        }
    }
}

impl<L> egg::FromOp for PartialLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = MetaLangError<L::Error>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "[placeholder]" | "[PLACEHOLDER]" | "[Placeholder]" => {
                if children.is_empty() {
                    Ok(Self::Placeholder)
                } else {
                    Err(MetaLangError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }

            _ => L::from_op(op, children)
                .map(Self::Finished)
                .map_err(MetaLangError::BadOp),
        }
    }
}

/// Tries to parse a list of tokens into a list of nodes in the language
///
/// # Errors
///
/// This function will return an error if it cannot be parsed as a partial term
pub fn partial_parse<L>(value: &[String]) -> Result<Vec<PartialLang<L>>, MetaLangError<L::Error>>
where
    L: FromOp + MetaInfo,
    L::Error: Display,
{
    let mut children_ids = VecDeque::new();
    let mut nodes = Vec::new();
    let mut greedy_children = Vec::new();
    for token in value {
        for n in 0.. {
            if let Ok(node) = PartialLang::<L>::from_op(token, greedy_children.clone()) {
                nodes.push(node);
                greedy_children.clear();
                children_ids.push_back(Id::from(nodes.len() - 1));
                break;
            }
            let next = children_ids.pop_front().unwrap_or_else(|| {
                nodes.push(PartialLang::Placeholder);
                Id::from(nodes.len() - 1)
            });
            greedy_children.push(next);

            if n > L::MAX_ARITY {
                return Err(MetaLangError::MaxArity(n));
            }
        }
        // }
    }
    Ok(nodes)
}

// pub fn partial_parse<L: FromOp + MetaInfo>(
//     mut token_list: Vec<String>,
// ) -> Result<Vec<Option<L>>, MetaLangError<E>> {
//     token_list.reverse();
//     let max_arity = 1233;

//     let mut children_ids = Vec::new();
//     let mut nodes = Vec::new();
//     for token in &token_list {
//         // Either we are parsing a parent, then we take all the children currently on the stack
//         if let Ok(node) = L::from_op(token, children_ids.clone()) {
//             nodes.push(Some(node));
//             children_ids.clear();
//             children_ids.push(Id::from(nodes.len() - 1));
//         // Or we are parsing a sibling child with no children, so we put it on the stack
//         } else if let Ok(node) = L::from_op(token, Vec::new()) {
//             nodes.push(Some(node));
//             children_ids.push(Id::from(nodes.len() - 1));
//         // Or we are parsing and incomplete parent that only has some children already generated
//         } else {
//             for n in 0..max_arity {
//                 if let Ok(node) = L::from_op(token, children_ids.clone()) {
//                     nodes.push(Some(node));
//                     children_ids.clear();
//                     children_ids.push(Id::from(nodes.len() - 1));
//                     break;
//                 }
//                 nodes.push(None);
//                 children_ids.push(Id::from(nodes.len() - 1));
//                 if n > max_arity {
//                     return Err(MetaLangError::MaxArity(n));
//                 }
//             }
//         }
//     }
//     Ok(nodes)
// }

#[cfg(test)]
mod tests {
    use crate::meta_lang::SketchLang;
    use crate::trs::{MetaInfo, halide::HalideLang};

    // use crate::typing::typecheck_expr;

    use super::*;

    #[test]
    fn parse_and_print() {
        let sketch = "(contains (max (min 1 [placeholder]) ?))"
            .parse::<PartialTerm<SketchLang<HalideLang>>>()
            .unwrap();
        assert_eq!(
            &sketch.to_string(),
            "(contains (max (min 1 [placeholder]) ?))"
        );
    }

    #[test]
    fn operators() {
        let known_operators = vec![
            "[placeholder]",
            "?",
            "contains",
            "or",
            "+",
            "-",
            "*",
            "/",
            "%",
            "max",
            "min",
            "<",
            ">",
            "!",
            "<=",
            ">=",
            "==",
            "!=",
            "||",
            "&&",
        ];
        assert_eq!(
            PartialLang::<SketchLang<HalideLang>>::operators(),
            known_operators
        );
    }
}
