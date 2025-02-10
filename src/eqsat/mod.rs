mod hooks;

pub mod conf;

use std::fmt::{Debug, Display};

use egg::{
    Analysis, CostFunction, EGraph, Extractor, Id, Iteration, Language, RecExpr, Report, Rewrite,
    Runner, SimpleScheduler,
};
use log::info;
use serde::Serialize;

use crate::sketch::{extract, Sketch};

pub use conf::{EqsatConf, EqsatConfBuilder};

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug)]
pub struct Eqsat<'a, L, N>
where
    L: Language + Display + 'static,
    N: Analysis<L> + Clone + Default + Debug + 'static,
    N::Data: Clone,
{
    conf: EqsatConf,
    start_material: StartMaterial<'a, L, N>,
    goals: Vec<RecExpr<L>>,
    rules: &'a [Rewrite<L, N>],
}

impl<'a, L, N> Eqsat<'a, L, N>
where
    L: Language + Display + 'static,
    N: Analysis<L> + Clone + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    /// Create a new Equality Saturation
    ///
    #[must_use]
    pub fn new(start_material: StartMaterial<'a, L, N>, rules: &'a [Rewrite<L, N>]) -> Self {
        Self {
            conf: EqsatConf::default(),
            goals: vec![],
            start_material,
            rules,
        }
    }

    /// With the following conf.
    #[must_use]
    pub fn with_conf(mut self, conf: EqsatConf) -> Self {
        self.conf = conf;
        self
    }

    /// With the following goals to check.
    #[must_use]
    pub fn with_goals(mut self, goals: Vec<RecExpr<L>>) -> Self {
        self.goals = goals;
        self
    }

    /// Runs single cycle of to prove an expression to be equal to the goals
    /// (most often true or false) with the given ruleset
    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn run(self) -> EqsatResult<L, N> {
        match &self.start_material {
            StartMaterial::RecExprs(exprs) => {
                let expr_strs = exprs
                    .iter()
                    .map(|x| x.to_string())
                    .reduce(|mut a, b| {
                        a.push_str(", ");
                        a.push_str(&b);
                        a
                    })
                    .expect("Eqsat needs at least one starting material!");
                info!("Running Eqsat with Expressions: [{expr_strs}]");
            }
            StartMaterial::EGraph { .. } => info!("Running Eqsat with previous EGraph"),
        }

        let mut runner = Runner::default()
            .with_iter_limit(self.conf.iter_limit)
            .with_node_limit(self.conf.node_limit)
            .with_memory_limit(self.conf.memory_limit)
            .with_time_limit(self.conf.time_limit);

        if self.conf.explanation {
            info!("Running with explanations");
            runner = runner.with_explanations_enabled();
        };

        if self.conf.root_check {
            info!("Installing root_check hook");
            runner = runner.with_hook(hooks::root_check_hook());
        }

        if self.conf.memory_log {
            info!("Installing memory display hook");
            runner = runner.with_hook(hooks::memory_hook());
        }

        if !self.goals.is_empty() {
            info!("Installing goals check hook");
            runner = runner.with_hook(hooks::goals_check_hook(self.goals));
        }

        // TODO ADD CONFIG OPTION
        runner = runner.with_scheduler(SimpleScheduler);

        let egraph_roots = match self.start_material {
            StartMaterial::RecExprs(vec) => {
                for expr in vec {
                    runner = runner.with_expr(expr);
                }
                None
            }
            StartMaterial::EGraph { egraph, roots } => {
                runner = runner.with_egraph(*egraph);
                Some(roots)
            }
        };

        runner = runner.run(self.rules);

        let report = runner.report();

        // Workaround since we need to deconstruct the starting material
        let roots = match egraph_roots {
            Some(roots) => roots,
            None => runner.roots,
        };

        info!("{}", &report);
        EqsatResult {
            runner_args: self.conf.clone(),
            egraph: runner.egraph,
            iterations: runner.iterations,
            roots,
            report,
        }
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
    runner_args: EqsatConf,
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
        CF::Cost: Ord,
    {
        extract::eclass_extract(sketch, cost_fn, &self.egraph, root).unwrap()
    }

    /// Extract
    pub fn satisfies_sketch(&self, root_index: usize, sketch: &Sketch<L>) -> bool {
        let root = self.roots[root_index];
        extract::eclass_satisfies_sketch(sketch, &self.egraph, root)
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
    RecExprs(&'a [&'a RecExpr<L>]),
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
