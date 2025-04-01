// The whole folder is in large parts Copy-Paste from https://github.com/Bastacyclop/egg-sketches/blob/main/src/sketch.rs
// Thank you very much for that!

use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{FromOp, Id, Language, RecExpr};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::MetaLangError;
use crate::trs::SymbolType::MetaSymbol;
use crate::trs::{MetaInfo, SymbolInfo};

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
    let mut children_ids = Vec::new();
    let mut nodes = Vec::new();
    for token in value {
        // Sibling case
        let node = PartialLang::<L>::from_op(token, vec![]).or_else(|_| {
            // Parent case (has to take all the existing children_ids)
            loop {
                if let Ok(node) = PartialLang::<L>::from_op(token, children_ids.clone()) {
                    children_ids.clear();
                    break Ok(node);
                }
                nodes.push(PartialLang::Placeholder);
                children_ids.push(Id::from(nodes.len() - 1));

                if children_ids.len() > L::MAX_ARITY {
                    break Err(MetaLangError::MaxArity(
                        token.to_owned(),
                        children_ids.len(),
                    ));
                }
            }
        })?;
        nodes.push(node);
        children_ids.push(Id::from(nodes.len() - 1));
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta_lang::SketchLang;
    use crate::trs::{MetaInfo, halide::HalideLang};

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
