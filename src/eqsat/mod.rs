mod hooks;
mod scheduler;

pub mod conf;

use std::fmt::{Debug, Display};

use egg::{
    Analysis, CostFunction, EGraph, Extractor, Id, Iteration, Language, RecExpr, Report, Rewrite,
    RewriteScheduler, Runner,
};
use log::info;
use serde::Serialize;

use crate::eqsat::hooks::targe_hook;
use crate::sketch::{self, Sketch};

pub use conf::{EqsatConf, EqsatConfBuilder};
pub use scheduler::BudgetScheduler;

/// Runs single cycle of to prove an expression to be equal to the target
/// (most often true or false) with the given ruleset
#[expect(clippy::missing_panics_doc)]
#[must_use]
pub fn eqsat<L, N, S>(
    conf: &EqsatConf,
    start_material: StartMaterial<L, N>,
    rules: &[Rewrite<L, N>],
    target: Option<RecExpr<L>>,
    scheduler: S,
) -> EqsatResult<L, N>
where
    L: Language + Display + 'static,
    S: RewriteScheduler<L, N> + 'static,
    N: Analysis<L> + Clone + Default + Debug + 'static,
    N::Data: Clone + Serialize,
{
    let mut runner = Runner::default()
        .with_scheduler(scheduler)
        .with_iter_limit(conf.iter_limit)
        .with_node_limit(conf.node_limit)
        .with_time_limit(conf.time_limit);

    if conf.explanation {
        info!("Running with explanations");
        runner = runner.with_explanations_enabled();
    }

    if conf.root_check {
        info!("Installing root_check hook");
        runner = runner.with_hook(hooks::root_check_hook());
    }

    if conf.memory_log {
        info!("Installing memory display hook");
        runner = runner.with_hook(hooks::memory_log_hook());
    }

    if let Some(t) = target {
        info!("Adding target");
        runner = runner.with_hook(targe_hook(t));
    }

    let egraph_roots = match start_material {
        StartMaterial::RecExprs(vec) => {
            assert!(
                (!vec.is_empty()),
                "Eqsat needs at least one starting material!"
            );
            let expr_strs: Vec<_> = vec.iter().map(|x| x.to_string()).collect();
            info!("Running Eqsat with Expressions: {expr_strs:?}");
            for expr in vec {
                runner = runner.with_expr(expr);
            }
            None
        }
        StartMaterial::EGraph { egraph, roots } => {
            info!("Running Eqsat with previous EGraph");
            runner = runner.with_egraph(*egraph);
            Some(roots)
        }
    };

    runner = runner.run(rules);

    let report = runner.report();
    info!("{}", &report);
    EqsatResult {
        egraph: runner.egraph,
        iterations: runner.iterations,
        roots: egraph_roots.unwrap_or(runner.roots),
        report,
    }
}

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug, Serialize)]
pub struct EqsatResult<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Serialize + Clone,
{
    // stats_history: Vec<EqsatStats>,
    egraph: EGraph<L, N>,
    iterations: Vec<Iteration<()>>,
    roots: Vec<Id>,
    report: Report,
}

impl<L, N> EqsatResult<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Serialize + Clone,
{
    // Extract via a classic cost function
    pub fn classic_extract<CF>(&self, root: Id, cost_fn: CF) -> (CF::Cost, RecExpr<L>)
    where
        CF: CostFunction<L>,
    {
        let extractor = Extractor::new(&self.egraph, cost_fn);
        extractor.find_best(root)
    }

    /// Extract with a sketch
    #[expect(clippy::missing_panics_doc)]
    pub fn sketch_extract<CF>(
        &self,
        root: Id,
        cost_fn: CF,
        sketch: &Sketch<L>,
    ) -> (CF::Cost, RecExpr<L>)
    where
        CF: CostFunction<L> + Debug,
        CF::Cost: Ord + 'static,
    {
        sketch::eclass_extract(sketch, cost_fn, &self.egraph, root).unwrap()
    }

    /// Check if sketch is satisfied
    pub fn satisfies_sketch(&self, root_index: usize, sketch: &Sketch<L>) -> bool {
        let root = self.roots[root_index];
        sketch::eclass_contains(sketch, &self.egraph, root)
    }

    /// Returns the root `egg::Id`
    ///
    /// Warning: Those are not necessarily canonical!
    pub fn roots(&self) -> &[Id] {
        &self.roots
    }

    pub fn report(&self) -> &Report {
        &self.report
    }

    pub fn iterations(&self) -> &[Iteration<()>] {
        &self.iterations
    }

    pub fn egraph(&self) -> &EGraph<L, N> {
        &self.egraph
    }

    pub fn egraph_mut(&mut self) -> &mut EGraph<L, N> {
        &mut self.egraph
    }
}

#[derive(Clone, Debug)]
pub enum StartMaterial<'a, L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Clone,
{
    RecExprs(Vec<&'a RecExpr<L>>),
    EGraph {
        egraph: Box<EGraph<L, N>>,
        roots: Vec<Id>,
    },
}

impl<L, N> From<EqsatResult<L, N>> for StartMaterial<'_, L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    fn from(eqsat_result: EqsatResult<L, N>) -> Self {
        StartMaterial::EGraph {
            egraph: Box::new(eqsat_result.egraph),
            roots: eqsat_result.roots,
        }
    }
}

impl<'a, L, N> From<&'a RecExpr<L>> for StartMaterial<'a, L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    fn from(rec_expr: &'a RecExpr<L>) -> StartMaterial<'a, L, N> {
        StartMaterial::RecExprs(vec![rec_expr])
    }
}

impl<'a, L, N> From<Vec<&'a RecExpr<L>>> for StartMaterial<'a, L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    fn from(rec_exprs: Vec<&'a RecExpr<L>>) -> StartMaterial<'a, L, N> {
        StartMaterial::RecExprs(rec_exprs)
    }
}
