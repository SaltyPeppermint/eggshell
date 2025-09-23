mod comparison;

use std::fmt::{Debug, Display, Formatter};
use std::mem::Discriminant;

use egg::{Id, Language, RecExpr};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

pub use comparison::{FirstErrorDistance, compare};

pub type ProbabilisticRecExpr<L> = RecExpr<ProbabilisticLang<L>>;

#[derive(Debug, PartialEq, Clone, Hash, Ord, Eq, PartialOrd, Serialize, Deserialize)]
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

    pub fn lower(higher: &RecExpr<Self>) -> RecExpr<L> {
        higher
            .into_iter()
            .map(|partial_node| partial_node.inner().to_owned())
            .collect::<Vec<_>>()
            .into()
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

impl<L> egg::FromOp for ProbabilisticLang<L>
where
    L::Error: Display,
    L: egg::FromOp,
{
    type Error = L::Error;

    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        Ok(ProbabilisticLang::NoProb(L::from_op(op, children)?))
    }
}
