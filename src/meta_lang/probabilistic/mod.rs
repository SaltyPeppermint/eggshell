mod comparison;

use std::fmt::{Debug, Display, Formatter};
use std::mem::Discriminant;

use egg::{Id, Language, RecExpr};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants};

use crate::trs::{LangExtras, SymbolInfo};

pub use comparison::FirstErrorDistance;

pub type ProbabilisticRecExpr<L> = RecExpr<ProbabilisticLang<L>>;

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
pub enum ProbabilisticLang<L: Language> {
    NoProb(L),
    WithProb { inner: L, prob: OrderedFloat<f64> },
}

impl<L: Language> ProbabilisticLang<L> {
    pub fn new(inner: L, prob: Option<f64>) -> Self {
        match prob {
            Some(f) => Self::WithProb {
                inner: inner,
                prob: f.into(),
            },
            None => Self::NoProb(inner),
        }
    }

    pub fn inner(&self) -> &L {
        match self {
            Self::NoProb(inner) => inner,
            Self::WithProb { inner, .. } => inner,
        }
    }

    pub fn inner_mut(&mut self) -> &mut L {
        match self {
            Self::NoProb(inner) => inner,
            Self::WithProb { inner, .. } => inner,
        }
    }

    pub fn prob(&self) -> Option<f64> {
        match self {
            ProbabilisticLang::NoProb(_) => None,
            ProbabilisticLang::WithProb { prob, .. } => Some((*prob).into()),
        }
    }
}

impl<L: Language> Language for ProbabilisticLang<L> {
    type Discriminant = (Discriminant<Self>, <L as Language>::Discriminant);

    fn discriminant(&self) -> Self::Discriminant {
        let discr = std::mem::discriminant(self);
        (discr, self.inner().discriminant())
    }

    fn matches(&self, other: &Self) -> bool {
        self.discriminant() == other.discriminant()
    }

    fn children(&self) -> &[Id] {
        self.inner().children()
    }

    fn children_mut(&mut self) -> &mut [Id] {
        self.inner_mut().children_mut()
    }
}

impl<L: Language + LangExtras> LangExtras for ProbabilisticLang<L> {
    fn symbol_info(&self) -> SymbolInfo {
        self.inner().symbol_info()
    }

    fn operators() -> Vec<&'static str> {
        L::operators()
    }

    fn pretty_string(&self) -> String {
        match self {
            ProbabilisticLang::WithProb { inner, prob } => {
                format!("{inner}\n{prob}")
            }
            ProbabilisticLang::NoProb(inner) => inner.to_string(),
        }
    }

    const NUM_SYMBOLS: usize = L::NUM_SYMBOLS + Self::COUNT;

    const MAX_ARITY: usize = L::MAX_ARITY;
}

impl<L: Language + Display> Display for ProbabilisticLang<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbabilisticLang::WithProb { inner, prob } => {
                write!(f, "{inner}: {prob}")
            }
            ProbabilisticLang::NoProb(inner) => write!(f, "{inner}"),
        }
    }
}
