pub mod results;
pub(crate) mod utils;

use std::time::Duration;

use colored::Colorize;
use egg::{EGraph, Pattern, RecExpr, Rewrite, StopReason};
use log::info;
use serde::Serialize;

use crate::errors::EggShellError;
use crate::trs::Trs;
use results::{EqsatReport, EqsatResult, EqsatStats};
use utils::RunnerArgs;

pub(crate) type ClassId = egg::Id;

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug, Serialize)]
pub struct Eqsat<R>
where
    R: Trs,
{
    index: usize,
    phases_limit: Option<usize>,
    time_limit: Option<Duration>,
    phase: usize,
    total_time: Duration,
    runner_args: RunnerArgs,
    stats_history: Vec<EqsatStats>,
    last_egraph: Option<EGraph<R::Language, R::Analysis>>,
    last_roots: Option<Vec<ClassId>>,
}

impl<R> Eqsat<R>
where
    R: Trs,
{
    /// Create a new Equality Saturation
    /// Is generic over a given [`Trs`]
    ///
    /// # Errors
    ///
    /// Will return an error if the starting term is not parsable in the
    /// [`Trs::Language`].
    pub fn new(index: usize) -> Result<Self, EggShellError> {
        Ok(Self {
            index,
            phases_limit: Some(10),
            time_limit: None,
            phase: 0,
            total_time: Duration::default(),
            runner_args: RunnerArgs::default(),
            last_egraph: None,
            last_roots: None,
            stats_history: Vec::new(),
        })
    }

    /// With the maximum number of phases.
    /// Defaults is 10
    #[must_use]
    pub fn with_phase_limit(mut self, phase_limit: Option<usize>) -> Self {
        self.phases_limit = phase_limit;
        self
    }

    /// With the maximum number of phases.
    /// Defaults is 10
    #[must_use]
    pub fn with_time_limit(mut self, time_limit: Option<Duration>) -> Self {
        self.time_limit = time_limit;
        self
    }

    /// With the runner parameters.
    #[must_use]
    pub fn with_runner_args(mut self, runner_params: RunnerArgs) -> Self {
        self.runner_args = runner_params;
        self
    }

    pub fn limit_reached(&self) -> bool {
        match (self.time_limit, self.phases_limit) {
            (None, None) => false,
            (None, Some(phase)) => self.phase >= phase,
            (Some(time), None) => self.total_time >= time,
            (Some(time), Some(phase)) => self.phase >= phase || self.total_time >= time,
        }
    }

    /// Run one loop of Equality Saturation to prove an expression is equal to a number
    /// of goals
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn run_goal_once(
        &mut self,
        start_expr: &RecExpr<R::Language>,
        goals: &[Pattern<R::Language>],
    ) -> EqsatResult<R::Language, R::Analysis> {
        let ruleset = R::maximum_ruleset();
        let rules = R::rules(&ruleset);
        let EqsatReport {
            egraph,
            roots,
            stats,
        } = self.single_eqsat(self.phase, start_expr, &rules);

        if let Some(expr) = utils::check_solved(goals, &egraph, *roots.last().unwrap()) {
            self.stats_history.push(stats);
            info!("{}", "Solved:".bright_green().bold());
            info!("{expr}");
            return EqsatResult::Solved(expr);
        }
        if matches!(stats.stop_reason, StopReason::Saturated) {
            self.stats_history.push(stats);
            info!("{}", "Showed to be undecidable!".bright_green().bold());
            return EqsatResult::Undecidable;
        }

        self.stats_history.push(stats);

        self.last_egraph = Some(egraph);
        self.last_roots = Some(roots);
        info!("{}", "Ran out of Ressources:".bright_green().bold());
        info!("Nodes: {}", self.stats_history.last().unwrap().egraph_size);
        info!(
            "Iterations {}",
            self.stats_history.last().unwrap().iterations
        );
        EqsatResult::LimitReached(Box::new(self.last_egraph.clone().unwrap()))
    }

    /// Run one loop of Equality Saturation to prove an expression is equal to a number
    /// of goals
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn run_simplify_once(
        &mut self,
        start_expr: &RecExpr<R::Language>,
    ) -> EGraph<R::Language, R::Analysis> {
        let ruleset = R::maximum_ruleset();
        let rules = R::rules(&ruleset);
        // if we simplify, we have no goals obviously
        let EqsatReport {
            egraph,
            roots,
            stats,
            ..
        } = self.single_eqsat(self.phase, start_expr, &rules);
        self.stats_history.push(stats);

        self.last_egraph = Some(egraph.clone());
        self.last_roots = Some(roots);
        info!("{}", "Ran out of Ressources:".bright_green().bold());
        info!("Nodes: {}", self.stats_history.last().unwrap().egraph_size);
        info!(
            "Iterations {}",
            self.stats_history.last().unwrap().iterations
        );
        egraph
    }

    /// Runs single cycle of to prove an expression to be equal to the goals
    /// (most often true or false) with the given ruleset
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    fn single_eqsat(
        &self,
        phase: usize,
        start_expr: &RecExpr<R::Language>,
        rules: &[Rewrite<R::Language, R::Analysis>],
    ) -> EqsatReport<R> {
        // Set up the goals we will check for.
        let mut total_time = self.total_time;

        println!("====================================");
        println!("Simplifying expression:");
        println!("{start_expr}");

        let runner = utils::build_runner(&self.runner_args, start_expr).run(rules.iter());

        let exec_time: f64 = runner.iterations.iter().map(|i| i.total_time).sum();
        total_time += Duration::from_secs_f64(exec_time);

        info!("{}", runner.report());

        let stats = EqsatStats::new(
            self.index,
            start_expr.to_string(),
            runner.iterations.len(),
            runner.egraph.total_number_of_nodes(),
            runner.iterations.iter().map(|i| i.n_rebuilds).sum(),
            phase + 1,
            total_time,
            runner.stop_reason.unwrap(),
        );
        EqsatReport::new(runner.egraph, runner.roots, stats)
    }

    pub fn stats_history(&self) -> &[EqsatStats] {
        &self.stats_history
    }

    pub fn total_time(&self) -> Duration {
        self.total_time
    }

    pub fn last_egraph(&self) -> Option<&EGraph<R::Language, R::Analysis>> {
        self.last_egraph.as_ref()
    }

    pub fn last_roots(&self) -> Option<&Vec<ClassId>> {
        self.last_roots.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::trs::halide::{Halide, Math};

    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_stmt: RecExpr<Math> = "( == 0 0 )".parse().unwrap();
        let goals = Halide::prove_goals();
        let mut eqsat = Eqsat::<Halide>::new(0).unwrap();
        let result = eqsat.run_goal_once(&false_stmt, &goals);
        assert_eq!("1", result.stringify_solved());
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_stmt: RecExpr<Math> = "( == 1 0 )".parse().unwrap();
        let goals = Halide::prove_goals();
        let mut eqsat = Eqsat::<Halide>::new(0).unwrap();
        let result = eqsat.run_goal_once(&false_stmt, &goals);
        assert_eq!("0", result.stringify_solved());
    }

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt: RecExpr<Math> = "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )".parse().unwrap();
        let goals = Halide::prove_goals();

        let mut eqsat = Eqsat::<Halide>::new(0).unwrap();
        let result = eqsat.run_goal_once(&true_stmt, &goals);
        assert_eq!("1", result.stringify_solved());
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt: RecExpr<Math> = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap();
        let goals = Halide::prove_goals();

        let mut eqsat = Eqsat::<Halide>::new(0).unwrap();
        let result = eqsat.run_goal_once(&false_stmt, &goals);
        assert_eq!("0", result.stringify_solved());
    }
}
