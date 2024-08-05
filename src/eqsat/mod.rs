pub(crate) mod utils;

use std::marker::PhantomData;

use egg::{CostFunction, EGraph, Extractor, Id, RecExpr, Report, Rewrite};
use log::info;
use serde::Serialize;

use crate::sketches::SketchNode;
use crate::trs::Trs;
use utils::RunnerArgs;

pub trait State {}

#[derive(Clone, Debug, Serialize)]
pub struct New;

impl State for New {}

#[derive(Clone, Debug, Serialize)]
pub struct Finished;

impl State for Finished {}

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug, Serialize)]
pub struct Eqsat<R, S>
where
    R: Trs,
    S: State,
{
    _state: PhantomData<S>,
    index: usize,
    runner_args: RunnerArgs,
    // stats_history: Vec<EqsatStats>,
    egraph: Option<EGraph<R::Language, R::Analysis>>,
    roots: Option<Vec<Id>>,
    report: Option<Report>,
}

impl<R, S> Eqsat<R, S>
where
    R: Trs,
    S: State,
{
    pub fn runner_args(&self) -> &RunnerArgs {
        &self.runner_args
    }
}

impl<R> Eqsat<R, New>
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
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            _state: PhantomData,
            index,
            runner_args: RunnerArgs::default(),
            egraph: None,
            roots: None,
            report: None,
        }
    }

    /// With the runner parameters.
    #[must_use]
    pub fn with_runner_args(mut self, runner_args: RunnerArgs) -> Self {
        self.runner_args = runner_args;
        self
    }

    /// Runs single cycle of to prove an expression to be equal to the goals
    /// (most often true or false) with the given ruleset
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn run(
        &self,
        start_expr: &RecExpr<R::Language>,
        rules: &[Rewrite<R::Language, R::Analysis>],
    ) -> Eqsat<R, Finished> {
        println!("====================================");
        println!("Running with Expression:");
        println!("{start_expr}");

        let runner = utils::build_runner(&self.runner_args, start_expr).run(rules.iter());

        let report = runner.report();
        info!("{}", &report);
        Eqsat {
            _state: PhantomData,
            index: self.index,
            runner_args: self.runner_args.clone(),
            egraph: Some(runner.egraph),
            roots: Some(runner.roots),
            report: Some(report),
        }
    }
}

impl<R> Eqsat<R, Finished>
where
    R: Trs,
{
    // Extract
    #[allow(clippy::missing_panics_doc)]
    pub fn classic_extract<CF>(&self, cost_fn: CF) -> Vec<(CF::Cost, RecExpr<R::Language>)>
    where
        CF: CostFunction<R::Language>,
    {
        let egraph = self.egraph.as_ref().unwrap();
        let extractor = Extractor::new(egraph, cost_fn);
        self.roots
            .as_ref()
            .unwrap()
            .iter()
            .map(|root| extractor.find_best(*root))
            .collect()
    }

    // Extract
    #[allow(clippy::missing_panics_doc)]
    pub fn sketch_extract<CF>(
        &self,
        cost_fn: CF,
        sketches: &[SketchNode<R::Language>],
    ) -> Vec<(CF::Cost, RecExpr<R::Language>)>
    where
        CF: CostFunction<R::Language>,
    {
        let egraph = self.egraph.as_ref().unwrap();
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        trs::halide::{Halide, MathEquations},
        utils::AstSize2,
    };

    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_stmt: RecExpr<MathEquations> = "( == 0 0 )".parse().unwrap();
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide, _>::new(0);
        let result = eqsat.run(&false_stmt, &rules);
        let extracted = result.classic_extract(AstSize2);
        assert_eq!("1", extracted[0].1.to_string());
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_stmt: RecExpr<MathEquations> = "( == 1 0 )".parse().unwrap();
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide, _>::new(0);
        let result = eqsat.run(&false_stmt, &rules);
        let extracted = result.classic_extract(AstSize2);
        assert_eq!("0", extracted[0].1.to_string());
    }

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt: RecExpr<MathEquations> = "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )".parse().unwrap();
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide, _>::new(0);
        let result = eqsat.run(&true_stmt, &rules);
        let extracted = result.classic_extract(AstSize2);
        assert_eq!("1", extracted[0].1.to_string());
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt: RecExpr<MathEquations> = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap();
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide, _>::new(0);
        let result = eqsat.run(&false_stmt, &rules);
        let extracted = result.classic_extract(AstSize2);
        assert_eq!("0", extracted[0].1.to_string());
    }
}
