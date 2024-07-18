use std::time::Duration;

use egg::{Analysis, Language, RecExpr, Runner};
use pyo3::prelude::*;
use serde::Serialize;

/// Struct to hold the arguments with which the [`egg::Runner`] is set up
#[pyclass]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunnerArgs {
    pub iter: Option<usize>,
    pub nodes: Option<usize>,
    pub time: Option<Duration>,
}

#[pymethods]
impl RunnerArgs {
    #[must_use]
    #[new]
    pub(crate) fn new(iter: Option<usize>, nodes: Option<usize>, time: Option<Duration>) -> Self {
        Self { iter, nodes, time }
    }
}

impl Default for RunnerArgs {
    #[must_use]
    fn default() -> Self {
        Self {
            iter: Some(10_000_000),
            nodes: Some(100_000),
            // Do we actually want a time limit?
            time: Some(Duration::new(10, 0)),
        }
    }
}

#[must_use]
pub(crate) fn build_runner<L, N>(runner_params: &RunnerArgs, expr: &RecExpr<L>) -> Runner<L, N>
where
    L: Language,
    N: Analysis<L> + Default,
{
    // Initialize a simple runner and run it.
    let mut runner = Runner::default();
    if let Some(iter_limit) = runner_params.iter {
        runner = runner.with_iter_limit(iter_limit);
    };
    if let Some(node_limit) = runner_params.nodes {
        runner = runner.with_node_limit(node_limit);
    };

    if let Some(time_limit) = runner_params.time {
        runner = runner.with_time_limit(time_limit);
    };
    runner.with_expr(expr)
}

// #[must_use]
// pub(crate) fn check_solved<L, N>(
//     goals: &[Pattern<L>],
//     egraph: &EGraph<L, N>,
//     id: Id,
// ) -> Option<String>
// where
//     L: Language + Display,
//     N: Analysis<L>,
// {
//     for goal in goals {
//         if (goal.search_eclass(egraph, id)).is_some() {
//             return Some(goal.ast.to_string());
//         }
//     }
//     None
// }
