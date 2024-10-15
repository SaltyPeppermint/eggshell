use std::time::Duration;

use bon::Builder;
use egg::{Analysis, Language, RecExpr, Runner, SimpleScheduler};
use hashbrown::HashSet;
use log::{info, warn};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the arguments with which the [`egg::Runner`] is set up
#[expect(clippy::unsafe_derive_deserialize)]
#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Builder, Default)]
pub struct EqsatConf {
    #[builder(default = 1000)]
    pub iter_limit: usize,
    #[builder(default = 1_000_000_000)]
    pub node_limit: usize,
    #[builder(default = 32_000_000_000)]
    pub memory_limit: usize,
    #[builder(default = Duration::from_secs_f64(60.0))]
    pub time_limit: Duration,
    #[builder(default = false)]
    pub explanation: bool,
    #[builder(default = false)]
    pub root_check: bool,
    #[builder(default = false)]
    pub memory_log: bool,
}

#[pymethods]
impl EqsatConf {
    #[must_use]
    #[new]
    #[pyo3(signature = (explanation=false,root_check=false, memory_log=false, iter_limit=1000, node_limit=1_000_000_000, memory_limit=32_000_000_000, time_limit=60.0))]
    pub fn new(
        explanation: bool,
        root_check: bool,
        memory_log: bool,
        iter_limit: usize,
        node_limit: usize,
        memory_limit: usize,
        time_limit: f64,
    ) -> Self {
        Self {
            iter_limit,
            node_limit,
            memory_limit,
            time_limit: Duration::from_secs_f64(time_limit),
            explanation,
            root_check,
            memory_log,
        }
    }
}

#[must_use]
pub(crate) fn build_runner<L, N>(
    conf: &EqsatConf,
    // root_check: bool,
    exprs: &[RecExpr<L>],
) -> Runner<L, N>
where
    L: Language,
    N: Analysis<L> + Default,
{
    // Initialize a simple runner and run it.
    let mut runner = Runner::default()
        .with_iter_limit(conf.iter_limit)
        .with_node_limit(conf.node_limit)
        .with_memory_limit(conf.memory_limit)
        .with_time_limit(conf.time_limit);

    if conf.explanation {
        info!("Running with explanations");
        runner = runner.with_explanations_enabled();
    };

    if conf.root_check {
        info!("Installing root_check hook");
        runner = runner.with_hook(craft_root_check_hook());
    }

    if conf.memory_log {
        info!("Installing memory display hook");
        runner = runner.with_hook(craft_memory_hook());
    }

    // TODO ADD CONFIG OPTION
    runner = runner.with_scheduler(SimpleScheduler);

    if conf.root_check {
        runner = runner.with_hook(craft_root_check_hook());
    }

    for expr in exprs {
        runner = runner.with_expr(expr);
    }
    runner
}

fn craft_root_check_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String> + 'static
where
    L: Language,
    N: Analysis<L> + Default,
{
    move |r: &mut Runner<L, N>| {
        let mut uniq = HashSet::new();
        if r.roots.iter().all(|x| uniq.insert(*x)) {
            Ok(())
        } else {
            Err("Duplicate in roots".into())
        }
    }
}

#[expect(clippy::cast_precision_loss)]
fn craft_memory_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String> + 'static
where
    L: Language,
    N: Analysis<L> + Default,
{
    let contents = std::fs::read_to_string("/proc/meminfo").expect("Could not read /proc/meminfo");
    let mem_info = contents
        .lines()
        .find(|line| line.starts_with("MemTotal"))
        .expect("Could not find MemTotal line");

    let system_memory = mem_info
        .split(' ')
        .rev()
        .nth(1)
        .expect("Found the size")
        .parse::<f64>()
        .expect("Memory must be number")
        / 1_000_000.0;

    info!("System memory is {system_memory:.3} GB");

    move |r: &mut Runner<L, N>| {
        println!("Current Nodes: {}", r.egraph.total_number_of_nodes());
        println!("Current Iteration: {}", r.iterations.len());
        if let Some(usage) = memory_stats::memory_stats() {
            println!(
                "Current physical memory usage: {:.3} GB / {system_memory:.3} GB",
                usage.physical_mem as f64 / 1_000_000_000.0
            );
            // println!("Current virtual memory usage: {}", usage.virtual_mem);
        } else {
            warn!("Couldn't get the current memory usage :(");
        }
        Ok(())
    }
}
