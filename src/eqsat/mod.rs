pub mod conf;
pub mod hooks;
mod scheduler;

use std::fmt::{Debug, Display};

use egg::{Analysis, EGraph, Id, Language, RecExpr, Rewrite, RewriteScheduler, Runner};
use log::info;

use crate::eqsat::hooks::targe_hook;

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
) -> (Runner<L, N>, Vec<Id>)
where
    L: Language + Display + 'static,
    S: RewriteScheduler<L, N> + 'static,
    N: Analysis<L> + Clone + Default + Debug + 'static,
    N::Data: Clone,
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

    let roots = egraph_roots.unwrap_or(runner.roots.clone());
    (runner, roots)
}

#[derive(Clone, Debug)]
pub enum StartMaterial<'a, L, N>
where
    L: Language,
    N: Analysis<L> + Default + Debug,
    N::Data: Clone,
{
    RecExprs(Vec<&'a RecExpr<L>>),
    EGraph {
        egraph: Box<EGraph<L, N>>,
        roots: Vec<Id>,
    },
}

impl<L, N> StartMaterial<'_, L, N>
where
    L: Language,
    N: Analysis<L> + Default + Debug,
    N::Data: Clone,
{
    pub fn from_egraph_and_roots(egraph: EGraph<L, N>, roots: Vec<Id>) -> Self {
        StartMaterial::EGraph {
            egraph: Box::new(egraph),
            roots,
        }
    }
}

impl<'a, L, N> From<&'a RecExpr<L>> for StartMaterial<'a, L, N>
where
    L: Language,
    N: Analysis<L> + Default + Debug,
    N::Data: Clone,
{
    fn from(rec_expr: &'a RecExpr<L>) -> StartMaterial<'a, L, N> {
        StartMaterial::RecExprs(vec![rec_expr])
    }
}

impl<'a, L, N> From<Vec<&'a RecExpr<L>>> for StartMaterial<'a, L, N>
where
    L: Language,
    N: Analysis<L> + Default + Debug,
    N::Data: Clone,
{
    fn from(rec_exprs: Vec<&'a RecExpr<L>>) -> StartMaterial<'a, L, N> {
        StartMaterial::RecExprs(rec_exprs)
    }
}

#[expect(missing_docs, clippy::missing_panics_doc)]
pub fn grow_egraph_until<L, N, S>(
    search_name: &str,
    egraph: EGraph<L, N>,
    rules: &[Rewrite<L, N>],
    mut satisfied: S,
) -> EGraph<L, N>
where
    S: FnMut(&mut Runner<L, N>) -> bool + 'static,
    L: Language,
    N: Analysis<L>,
    N: Default,
{
    let search_name_hook = search_name.to_owned();
    let runner = egg::Runner::default()
        .with_scheduler(egg::SimpleScheduler)
        .with_iter_limit(100)
        .with_node_limit(100_000_000)
        .with_time_limit(std::time::Duration::from_secs(5 * 60))
        .with_hook(move |runner| {
            let mut out_of_memory = false;
            // hook 0 <- nothing
            // iteration 0
            // hook 1 <- #0 size etc after iteration 0 + memory after iteration 0
            if let Some(it) = runner.iterations.last() {
                out_of_memory = iteration_stats(&search_name_hook, it, runner.iterations.len());
            }

            if satisfied(runner) {
                Err(String::from("Satisfied"))
            } else if out_of_memory {
                Err(String::from("Out of Memory"))
            } else {
                Ok(())
            }
        })
        .with_egraph(egraph)
        .run(rules);
    iteration_stats(
        search_name,
        runner.iterations.last().unwrap(),
        runner.iterations.len(),
    );
    runner.print_report();
    runner.egraph
}

// search name,
// iteration number,
// physical memory,
// virtual memory,
// e-graph nodes,
// e-graph classes,
// applied rules,
// total time,
// hook time,
// search time,
// apply time,
// rebuild time
fn iteration_stats(search_name: &str, it: &egg::Iteration<()>, it_number: usize) -> bool {
    let memory = memory_stats::memory_stats().expect("could not get current memory usage");
    let out_of_memory = memory.virtual_mem > 8_000_000_000;
    let found = match &it.stop_reason {
        Some(egg::StopReason::Other(s)) => s == "Satisfied",
        _ => false,
    };
    eprintln!(
        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
        search_name,
        it_number,
        memory.physical_mem,
        memory.virtual_mem,
        it.egraph_nodes,
        it.egraph_classes,
        it.applied.iter().map(|(_, &n)| n).sum::<usize>(),
        it.total_time,
        it.hook_time,
        it.search_time,
        it.apply_time,
        it.rebuild_time,
        found
    );
    out_of_memory
}
