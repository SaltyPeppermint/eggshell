pub mod utils;

use egg::{CostFunction, EGraph, Extractor, Id, RecExpr, Report, Rewrite};
use log::info;
use serde::Serialize;
use utils::EqsatConfBuilder;

use crate::sketch::extract;
use crate::sketch::Sketch;
use crate::trs::Trs;
use utils::EqsatConf;

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug, Serialize)]
pub struct Eqsat<R>
where
    R: Trs,
{
    runner_args: EqsatConf,
    start_exprs: Vec<RecExpr<R::Language>>,
}

impl<R> Eqsat<R>
where
    R: Trs,
{
    #[must_use]
    pub fn runner_args(&self) -> &EqsatConf {
        &self.runner_args
    }

    /// Create a new Equality Saturation
    /// Is generic over a given [`Trs`]
    ///
    /// # Errors
    ///
    /// Will return an error if the starting term is not parsable in the
    /// [`Trs::Language`].
    #[must_use]
    pub fn new(start_exprs: Vec<RecExpr<R::Language>>) -> Self {
        Self {
            runner_args: EqsatConfBuilder::default().build(),
            start_exprs,
        }
    }

    /// With the runner parameters.
    #[must_use]
    pub fn with_conf(mut self, runner_args: EqsatConf) -> Self {
        self.runner_args = runner_args;
        self
    }

    /// Runs single cycle of to prove an expression to be equal to the goals
    /// (most often true or false) with the given ruleset
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn run(
        &self,
        // start_exprs: &[RecExpr<R::Language>],
        rules: &[Rewrite<R::Language, R::Analysis>],
    ) -> EqsatResult<R> {
        // println!("====================================");
        // println!("Running with Expression:");

        let runner = utils::build_runner(&self.runner_args, &self.start_exprs).run(rules.iter());

        let report = runner.report();
        info!("{}", &report);
        EqsatResult {
            runner_args: self.runner_args.clone(),
            egraph: runner.egraph,
            roots: runner.roots,
            report,
        }
    }
}

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug, Serialize)]
pub struct EqsatResult<R>
where
    R: Trs,
{
    runner_args: EqsatConf,
    // stats_history: Vec<EqsatStats>,
    egraph: EGraph<R::Language, R::Analysis>,
    roots: Vec<Id>,
    report: Report,
}

impl<R> EqsatResult<R>
where
    R: Trs,
{
    // Extract
    #[allow(clippy::missing_panics_doc)]
    pub fn classic_extract<CF>(&self, root: Id, cost_fn: CF) -> (CF::Cost, RecExpr<R::Language>)
    where
        CF: CostFunction<R::Language>,
    {
        let extractor = Extractor::new(&self.egraph, cost_fn);
        extractor.find_best(root)
    }

    //Extract
    #[allow(clippy::missing_panics_doc)]
    pub fn sketch_extract<CF>(
        &self,
        root: Id,
        cost_fn: CF,
        sketch: &Sketch<R::Language>,
    ) -> (CF::Cost, RecExpr<R::Language>)
    where
        CF: CostFunction<R::Language>,
        CF::Cost: Ord,
    {
        extract::eclass_extract(sketch, cost_fn, &self.egraph, root).unwrap()
    }

    //Extract
    #[allow(clippy::missing_panics_doc)]
    pub fn satisfies_sketch(&self, root_index: usize, sketch: &Sketch<R::Language>) -> bool {
        let root = self.roots[root_index];
        extract::eclass_satisfies_sketch(sketch, &self.egraph, root)
    }

    pub fn roots(&self) -> &[Id] {
        &self.roots
    }

    pub fn report(&self) -> &Report {
        &self.report
    }

    pub fn egraph(&self) -> &EGraph<R::Language, R::Analysis> {
        &self.egraph
    }
}

#[cfg(test)]
mod tests {
    use crate::{trs::halide::Halide, utils::AstSize2};

    use super::*;

    #[test]
    fn basic_eqsat_solved_true() {
        let false_expr = vec!["( == 0 0 )".parse().unwrap()];
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!("1", term.to_string());
    }

    #[test]
    fn basic_eqsat_solved_false() {
        let false_expr = vec!["( == 1 0 )".parse().unwrap()];
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide>::new(false_expr);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!("0", term.to_string());
    }

    #[test]
    #[should_panic(expected = "Different leaves in eclass 1: {Constant(35), Constant(0)}")]
    fn panic_on_false_expr() {
        let expr = vec![
            "( < ( + ( + ( * v0 35 ) v1 ) 35 ) ( + ( * ( + v0 1 ) 35 ) v1 ) )"
                .parse()
                .unwrap(),
        ];
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide>::new(expr);
        let _ = eqsat.run(&rules);
    }
}
