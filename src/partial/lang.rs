use std::fmt::{Display, Formatter};
use std::mem::{Discriminant, discriminant};

use egg::{Id, Language};
use strum::{EnumCount, IntoEnumIterator};

use crate::trs::SymbolType;
use crate::trs::{MetaInfo, SymbolInfo};

use super::error::PartialError;
use super::{PartialLang, PartialLangDiscriminants};

impl<L: Language> Language for PartialLang<L> {
    type Discriminant = (Discriminant<Self>, Option<<L as Language>::Discriminant>);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = discriminant(self);
        match self {
            PartialLang::Finished(x) => (discr, Some(x.discriminant())),
            Self::Pad => (discr, None),
        }
    }

    fn matches(&self, _other: &Self) -> bool {
        panic!("Comparing sketches to each other does not make sense!")
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Finished(n) => n.children(),
            Self::Pad => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Finished(n) => n.children_mut(),
            Self::Pad => &mut [],
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
            SymbolInfo::new(position + L::NUM_SYMBOLS, SymbolType::MetaSymbol)
        }
    }

    fn operators() -> Vec<&'static str> {
        let mut s = vec!["<pad>"];
        s.extend(L::operators());
        s
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for PartialLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Finished(sketch_node) => write!(f, "{sketch_node}"),
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
                .map(Self::Finished)
                .map_err(PartialError::BadOp),
        }
    }
}
