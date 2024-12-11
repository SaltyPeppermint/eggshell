use std::fmt::Display;

use egg::{Analysis, Language, RecExpr, StopReason};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::cli::Cli;
use crate::eqsat::{EqsatConf, EqsatResult};
use crate::sampling::SampleConf;

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + Display> {
    start_expr: RecExpr<L>,
    sample_data: Vec<SampleData<L>>,
    random_goals: Vec<usize>,
    metadata: MetaData,
}

impl<L: Language + Display> DataEntry<L> {
    #[must_use]
    pub fn new(
        start_expr: RecExpr<L>,
        sample_data: Vec<SampleData<L>>,
        random_goals: Vec<usize>,

        metadata: MetaData,
    ) -> Self {
        Self {
            start_expr,
            sample_data,
            random_goals,
            metadata,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData {
    uuid: String,
    folder: String,
    cli: Cli,
    timestamp: i64,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

impl MetaData {
    #[must_use]
    pub fn new(
        uuid: String,
        folder: String,
        cli: Cli,
        timestamp: i64,
        sample_conf: SampleConf,
        eqsat_conf: EqsatConf,
    ) -> Self {
        Self {
            uuid,
            folder,
            cli,
            timestamp,
            sample_conf,
            eqsat_conf,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SampleData<L: Language + Display> {
    sample: RecExpr<L>,
    generation: usize,
    baseline: Option<HashMap<usize, EqsatStats>>,
}

impl<L: Language + Display> SampleData<L> {
    #[must_use]
    pub fn new(
        sample: RecExpr<L>,
        generation: usize,
        baseline: Option<HashMap<usize, EqsatStats>>,
    ) -> Self {
        Self {
            sample,
            generation,
            baseline,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct EqsatStats {
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}

impl EqsatStats {
    #[must_use]
    pub fn new(
        stop_reason: StopReason,
        total_time: f64,
        total_nodes: usize,
        total_iters: usize,
    ) -> Self {
        Self {
            stop_reason,
            total_time,
            total_nodes,
            total_iters,
        }
    }
}

impl<L, N> From<EqsatResult<L, N>> for EqsatStats
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Serialize + Clone,
{
    fn from(result: EqsatResult<L, N>) -> Self {
        EqsatStats::new(
            result.report().stop_reason.clone(),
            result.report().total_time,
            result.report().egraph_nodes,
            result.report().iterations,
        )
    }
}
