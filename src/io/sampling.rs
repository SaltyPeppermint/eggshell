use std::fmt::Display;

use egg::{Analysis, FromOp, Language, RecExpr, StopReason};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::cli::Cli;
use crate::eqsat::{EqsatConf, EqsatResult};
use crate::explanation::IntermediateTerms;

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + FromOp + Display> {
    start_expr: RecExpr<L>,
    sample_data: Vec<SampleData<L>>,
    baselines: Option<HashMap<usize, HashMap<usize, EqsatStats>>>,
    metadata: MetaData,
}

impl<L: Language + FromOp + Display> DataEntry<L> {
    #[must_use]
    pub fn new(
        start_expr: RecExpr<L>,
        sample_data: Vec<SampleData<L>>,
        baselines: Option<HashMap<usize, HashMap<usize, EqsatStats>>>,

        metadata: MetaData,
    ) -> Self {
        Self {
            start_expr,
            sample_data,
            baselines,
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

    eqsat_conf: EqsatConf,
    rules: Vec<String>,
}

impl MetaData {
    #[must_use]
    pub fn new(
        uuid: String,
        folder: String,
        cli: Cli,
        timestamp: i64,

        eqsat_conf: EqsatConf,
        rules: Vec<String>,
    ) -> Self {
        Self {
            uuid,
            folder,
            cli,
            timestamp,

            eqsat_conf,
            rules,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SampleData<L: Language + FromOp + Display> {
    sample: RecExpr<L>,
    generation: usize,
    explanation: Option<ExplanationData<L>>,
}

impl<L: Language + FromOp + Display> SampleData<L> {
    #[must_use]
    pub fn new(
        sample: RecExpr<L>,
        generation: usize,
        explanation: Option<ExplanationData<L>>,
    ) -> Self {
        Self {
            sample,
            generation,
            explanation,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct ExplanationData<L: Language + FromOp + Display> {
    flat_string: String,
    explanation_chain: Vec<IntermediateTerms<L>>,
}

impl<L: Language + FromOp + Display> ExplanationData<L> {
    #[must_use]
    pub fn new(flat_string: String, explanation_chain: Vec<IntermediateTerms<L>>) -> Self {
        Self {
            flat_string,
            explanation_chain,
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
