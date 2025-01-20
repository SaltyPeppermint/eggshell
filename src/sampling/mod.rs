mod choices;
pub mod strategy;

use std::fmt::{Debug, Display};

use egg::{Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use serde::Serialize;
use thiserror::Error;

use crate::eqsat::EqsatConf;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SampleError {
    #[error("Batchsize impossible: {0}")]
    BatchSizeError(usize),
    #[error("Can't convert a non-finished list of choices")]
    ChoiceError,
    #[error("Extraction not possible for this eclasses as the analysis gave no expression due to too low limit of {0}!")]
    LimitError(usize),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Sample<L: Language + Display + Send + Sync> {
    start_exprs: RecExpr<L>,
    samples: HashMap<Id, HashSet<RecExpr<L>>>,
    eqsat_conf: EqsatConf,
}
