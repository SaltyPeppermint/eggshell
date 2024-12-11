mod class_cost;
mod conf;
mod hooks;

use std::fmt::{Debug, Display};

use egg::{
    Analysis, CostFunction, EGraph, Extractor, Id, Iteration, Language, RecExpr, Report, Rewrite,
    Runner, SimpleScheduler,
};
use hashbrown::HashMap;
use log::info;
use serde::Serialize;

use crate::sketch::{extract, Sketch};
use class_cost::{ClassExtractor, LutCost};

pub use conf::{EqsatConf, EqsatConfBuilder};

/// API accessible struct holding the equality Saturation
#[derive(Clone, Debug)]
pub struct Eqsat<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Clone,
{
    conf: EqsatConf,
    start_material: StartMaterial<L, N>,
}

#[derive(Clone, Debug)]
pub enum StartMaterial<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Clone,
{
    RecExprs(Vec<RecExpr<L>>),
    EGraph {
        egraph: Box<EGraph<L, N>>,
        roots: Vec<Id>,
    },
}

impl<L, N> Eqsat<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone + Default + Debug + Default,
    N::Data: Serialize + Clone,
{
    #[must_use]
    pub fn runner_args(&self) -> &EqsatConf {
        &self.conf
    }

    /// Create a new Equality Saturation
    /// Is generic over a given [`Trs`]
    ///
    /// # Errors
    ///
    /// Will return an error if the starting expression is not parsable in the
    /// [`Trs::Language`].
    #[must_use]
    pub fn new(start_material: StartMaterial<L, N>) -> Self {
        Self {
            conf: EqsatConf::default(),
            start_material,
        }
    }

    // /// Create a new Equality Saturation with a given egraph
    // /// Is generic over a given [`Trs`]
    // ///
    // /// # Errors
    // ///
    // /// Will return an error if the starting expr is not parsable in the
    // /// [`Trs::Language`].
    // #[must_use]
    // pub fn new(start_exprs: Vec<RecExpr<R::Language>>) -> Self {
    //     Self {
    //         conf: EqsatConf::default(),
    //         start_exprs,
    //     }
    // }

    /// With the runner parameters.
    #[must_use]
    pub fn with_conf(mut self, conf: EqsatConf) -> Self {
        self.conf = conf;
        self
    }

    /// Runs single cycle of to prove an expression to be equal to the goals
    /// (most often true or false) with the given ruleset
    #[must_use]
    pub fn run(
        self,
        // start_exprs: &[RecExpr<R::Language>],
        rules: &[Rewrite<L, N>],
    ) -> EqsatResult<L, N> {
        match &self.start_material {
            StartMaterial::RecExprs(exprs) => info!("Running Eqsat with Expressions: {:?}", exprs),
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
            runner = runner.with_hook(hooks::craft_root_check_hook());
        }

        if self.conf.memory_log {
            info!("Installing memory display hook");
            runner = runner.with_hook(hooks::craft_memory_hook());
        }

        // TODO ADD CONFIG OPTION
        runner = runner.with_scheduler(SimpleScheduler);

        if self.conf.root_check {
            runner = runner.with_hook(hooks::craft_root_check_hook());
        }

        let egraph_roots = match self.start_material {
            StartMaterial::RecExprs(vec) => {
                for expr in &vec {
                    runner = runner.with_expr(expr);
                }
                None
            }
            StartMaterial::EGraph { egraph, roots } => {
                runner = runner.with_egraph(*egraph);
                Some(roots)
            }
        };

        runner = runner.run(rules);

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

    /// Extract with a table of costs
    pub fn table_extract(
        &self,
        root: Id,
        cost_table: HashMap<(Id, usize), f64>,
    ) -> (f64, RecExpr<L>) {
        let cost_function = LutCost::new(cost_table, &self.egraph);
        let extractor = ClassExtractor::new(&self.egraph, cost_function);
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
        CF::Cost: Ord + Debug,
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

impl<L, N> From<EqsatResult<L, N>> for StartMaterial<L, N>
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Serialize + Clone,
{
    fn from(eqsat_result: EqsatResult<L, N>) -> Self {
        StartMaterial::EGraph {
            egraph: Box::new(eqsat_result.egraph),
            roots: eqsat_result.roots,
        }
    }
}
