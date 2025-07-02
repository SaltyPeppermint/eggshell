mod choices;
pub mod sampler;

use std::fmt::{Debug, Display};

use egg::{Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use serde::Serialize;
use thiserror::Error;

pub use sampler::Sampler;

use crate::eqsat::EqsatConf;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SampleError {
    #[error("Can't convert a non-finished list of choices")]
    UnfinishedChoice,
    #[error("Extraction not possible as this eclass contains no terms of appropriate size {0}!")]
    SizeLimit(usize),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Sample<L: Language + Display + Send + Sync> {
    start_exprs: RecExpr<L>,
    samples: HashMap<Id, HashSet<RecExpr<L>>>,
    eqsat_conf: EqsatConf,
}
