use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{Id, Language};
use strum::{EnumCount, IntoEnumIterator};

use crate::trs::SymbolType;
use crate::trs::{LangExtras, SymbolInfo};

use super::error::PartialError;
use super::{PartialLang, PartialLangDiscriminants};

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialLang::Finished { inner, prob: _ } => (discr, Some(inner.discriminant())),
            Self::Pad => (discr, None),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }

    fn children(&self) -> &[Id] {
        match self {
            PartialLang::Finished { inner, prob: _ } => inner.children(),
            Self::Pad => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            PartialLang::Finished { inner, prob: _ } => inner.children_mut(),
            Self::Pad => &mut [],
        }
    }
}

impl<L: Language + LangExtras> LangExtras for PartialLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        if let PartialLang::Finished { inner, prob: _ } = self {
            inner.symbol_info()
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
            PartialLang::Finished {
                inner,
                prob: Some(p),
            } => format!("{inner}\n{p}"),
            PartialLang::Finished {
                inner: _,
                prob: None,
            }
            | PartialLang::Pad => self.to_string(),
        }
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PartialLang::Finished {
                inner,
                prob: Some(p),
            } => write!(f, "{inner}: {p}"),
            PartialLang::Finished { inner, prob: None } => write!(f, "{inner}"),
            Self::Pad => write!(f, "<pad>"),
        }
    }
}

impl<L> egg::FromOp for PartialLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = PartialError<L>;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "[pad]" | "[PAD]" | "[Pad]" | "<pad>" | "<PAD>" | "<Pad>" => {
                if children.is_empty() {
                    Ok(Self::Pad)
                } else {
                    Err(PartialError::BadChildren(egg::FromOpError::new(
                        op, children,
                    )))
                }
            }

            _ => L::from_op(op, children)
                .map(|l| Self::Finished {
                    inner: l,
                    prob: None,
                })
                .map_err(PartialError::BadOp),
        }
    }
}
