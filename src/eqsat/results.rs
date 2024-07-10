use std::{fmt::Display, time::Duration};

use egg::{Analysis, EGraph, Language, StopReason};
use pyo3::pyclass;
use serde::Serialize;

use crate::{io::structs::OtherSolverData, trs::Trs};

use super::ClassId;

/// A goal attempt can have 3 outcomes:
/// Solved, undecidable
/// or we ran out of ressources before we decide anything.
#[derive(Serialize, Debug, Clone)]
pub enum EqsatResult<L, N>
where
    L: Language + Serialize + Display,
    N: Analysis<L> + Serialize,
    N::Data: Serialize + Clone,
{
    Solved(String),
    Undecidable,
    LimitReached(Box<EGraph<L, N>>),
}

impl<L, N> EqsatResult<L, N>
where
    L: Language + Serialize + Display,
    N: Analysis<L> + Serialize,
    N::Data: Serialize + Clone,
{
    #[must_use]
    pub fn stringify_solved(self) -> String {
        match self {
            EqsatResult::Solved(solution) => solution,
            EqsatResult::Undecidable => String::from("UNDECIDEABLE!"),
            EqsatResult::LimitReached(egraph) => format!("{:#?}", egraph.dump()),
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
    pub stop_reason: StopReason,
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
        stop_reason: StopReason,
    ) -> Self {
        Self {
            index,
            start_expr,
            iterations,
            egraph_size,
            rebuilds,
            phases,
            time: total_time,
            stop_reason,
            condition: None,
            halide_data: None,
        }
    }
}

pub struct EqsatReport<R: Trs> {
    pub egraph: EGraph<R::Language, R::Analysis>,
    pub roots: Vec<ClassId>,
    pub stats: EqsatStats,
}

impl<R: Trs> EqsatReport<R> {
    pub(crate) fn new(
        egraph: EGraph<R::Language, R::Analysis>,
        roots: Vec<ClassId>,
        stats: EqsatStats,
    ) -> Self {
        Self {
            egraph,
            roots,
            stats,
        }
    }
}
