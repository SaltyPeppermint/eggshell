use std::fmt::Display;

use egg::{Language, RecExpr, StopReason};
use serde::{Deserialize, Serialize};

use crate::cli::Cli;
use crate::eqsat::EqsatConf;
use crate::sampling::SampleConf;

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + Display> {
    start_expr: RecExpr<L>,
    sample_data: Vec<SampleData<L>>,
    metadata: MetaData,
}

impl<L: Language + Display> DataEntry<L> {
    #[must_use]
    pub fn new(
        start_expr: RecExpr<L>,
        sample_data: Vec<SampleData<L>>,
        metadata: MetaData,
    ) -> Self {
        Self {
            start_expr,
            sample_data,
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
    baseline: Option<BaselineData>,
    explanation: Option<String>,
}

impl<L: Language + Display> SampleData<L> {
    #[must_use]
    pub fn new(
        sample: RecExpr<L>,
        generation: usize,
        baseline: Option<BaselineData>,
        explanation: Option<String>,
    ) -> Self {
        Self {
            sample,
            generation,
            baseline,
            explanation,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct BaselineData {
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}

impl BaselineData {
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
