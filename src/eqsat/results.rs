use std::time::Duration;

use egg::{EGraph, StopReason};
use pyo3::pyclass;
use serde::Serialize;

use crate::{io::structs::OtherSolverData, trs::Trs};

use super::ClassId;

/// Stringify the [`StopReason`] of the runner
#[must_use]
pub(crate) fn stringify_stop_reason(reason: &Option<StopReason>) -> String {
    match reason {
        Some(reason) => match reason {
            StopReason::Saturated => "Saturated!".into(),
            StopReason::IterationLimit(i) => format!("Reached iteration limit {i}"),
            StopReason::NodeLimit(i) => format!("Reached node limit {i}"),
            StopReason::TimeLimit(i) => format!("Reached time limit {i}"),
            StopReason::Other(i) => format!("Other Reason: {i}"),
        },
        None => String::new(),
    }
}

/// A goal attempt can have 3 outcomes:
/// Solved, undecidable
/// or we ran out of ressources before we decide anything.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub enum EqsatResult<T, U> {
    Solved(T),
    Undecidable,
    LimitReached(U),
}

impl<T: ToString, U> EqsatResult<T, U> {
    pub fn stringify_solved(self) -> EqsatResult<String, U> {
        match self {
            EqsatResult::Solved(x) => EqsatResult::Solved(x.to_string()),
            EqsatResult::Undecidable => todo!(),
            EqsatResult::LimitReached(x) => EqsatResult::LimitReached(x),
        }
    }
}

/// General stats about an equality saturation attempt.
#[pyclass]
#[derive(Serialize, Debug, Clone)]
pub struct EqsatStats {
    /// Index of the expression set to make debugging easier
    index: usize,
    /// The expression to be proved or simplified
    start_expr: String,
    /// Number of iterations used to prove the expression
    pub iterations: usize,
    /// The size of the egraph used to prove the expression
    pub egraph_size: usize,
    /// The number of rebuilds used to prove the expression
    rebuilds: usize,
    /// The number of phases used to prove the expression
    phases: usize,
    /// The time it took to prove the expression
    pub time: Duration,
    /// The reason the execution stopped
    runner_stop_reason: String,
    /// The condition of the rule
    condition: Option<String>,
    /// Halide Data for the expression
    halide_data: Option<OtherSolverData>,
}

impl EqsatStats {
    /// New [`EqsatStats`]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub(crate) fn new(
        index: usize,
        start_expr: String,
        iterations: usize,
        egraph_size: usize,
        rebuilds: usize,
        phases: usize,
        total_time: Duration,
        runner_stop_reason: String,
    ) -> Self {
        Self {
            index,
            start_expr,
            iterations,
            egraph_size,
            rebuilds,
            phases,
            time: total_time,
            runner_stop_reason,
            condition: None,
            halide_data: None,
        }
    }
}

pub struct EqsatReport<R: Trs> {
    pub egraph: EGraph<R::Language, R::Analysis>,
    pub roots: Vec<ClassId>,
    pub stats: EqsatStats,
    pub stop_reason: StopReason,
}

impl<R: Trs> EqsatReport<R> {
    pub(crate) fn new(
        egraph: EGraph<R::Language, R::Analysis>,
        roots: Vec<ClassId>,
        stats: EqsatStats,
        stop_reason: StopReason,
    ) -> Self {
        Self {
            egraph,
            roots,
            stats,
            stop_reason,
        }
    }
}
