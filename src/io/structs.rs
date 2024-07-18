use std::fmt::Display;
use std::time::Duration;

use egg::{CostFunction, Language, RecExpr};
use serde::Serialize;

use pyo3::prelude::*;

// use crate::eqsat::results::EqsatResult;
use crate::eqsat::utils::RunnerArgs;

/// Used to represent an expression
#[derive(Serialize, Debug)]
pub struct EqsatReport<L, C>
where
    L: Language + Display + Serialize,
    // N: Analysis<L> + Serialize,
    // N::Data: Serialize + Clone,
    C: CostFunction<L>,
    C::Cost: Serialize,
{
    pub index: usize,
    pub first_expr: RecExpr<L>,
    // pub stats_history: Vec<EqsatStats>,
    pub iteration_check: Option<bool>,
    pub total_time: Duration,
    pub runner_args: RunnerArgs,
    // Optional solver data, this is not really generic...
    pub other_solver_data: Option<OtherSolverData>,
    pub extracted_exprs: Vec<Vec<(C::Cost, RecExpr<L>)>>,
    // pub final_result: EqsatResult<L, N>,
}

/// Used to represent an expression
#[derive(Serialize, Debug, Clone, PartialEq)]
#[pyclass]
pub struct Expression {
    /// Index of the expression
    pub index: usize,
    /// the string of the expression
    pub term: String,
    /// Optional halide data
    pub other_solver: Option<OtherSolverData>,
}

/// Holds (optional) additional data of other solver results
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct OtherSolverData {
    /// Other Solvers's result for proving the expression
    pub result: String,
    /// The time it took the other solver to prove the expression
    pub time: f64,
}

#[pymethods]
impl Expression {
    #[allow(clippy::cast_possible_wrap)]
    #[must_use]
    #[getter]
    pub fn index(&self) -> i64 {
        self.index as i64
    }

    #[must_use]
    #[getter]
    pub fn term(&self) -> String {
        self.term.to_string()
    }
}

// /// Represents a [`Rule`] of the TRS
// #[derive(Serialize, Debug)]
// pub(crate) struct Rule {
//     // Index of the rule
//     pub index: usize,
//     // the LHS of the rule
//     pub lhs: String,
//     // the RHS of the rule
//     pub rhs: String,
//     // The condition to apply the rule
//     pub condition: Option<String>,
// }

// impl Rule {
//     #[must_use]
//     pub fn new(index: usize, lhs: String, rhs: String, condition: Option<String>) -> Self {
//         Self {
//             index,
//             lhs,
//             rhs,
//             condition,
//         }
//     }
// }
