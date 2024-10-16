use std::time::Duration;

use egg::{Analysis, Language, RecExpr, Runner, SimpleScheduler};
use hashbrown::HashSet;
use log::{info, warn};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the arguments with which the [`egg::Runner`] is set up
#[expect(clippy::unsafe_derive_deserialize)]
#[pyclass(frozen)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq)]
pub struct EqsatConf {
    pub iter_limit: Option<usize>,
    pub node_limit: Option<usize>,
    pub time_limit: Option<Duration>,
    pub explanation: bool,
    pub root_check: bool,
    pub memory_log: bool,
}

#[pymethods]
impl EqsatConf {
    #[must_use]
    #[new]
    #[pyo3(signature = (explanation=false,root_check=false, memory_log=false, iter_limit=Some(10_000_000), node_limit=Some(100_000), time_limit=Some(10.0)))]
    pub fn new(
        explanation: bool,
        root_check: bool,
        memory_log: bool,
        iter_limit: Option<usize>,
        node_limit: Option<usize>,
        time_limit: Option<f64>,
    ) -> Self {
        Self {
            iter_limit,
            node_limit,
            time_limit: time_limit.map(Duration::from_secs_f64),
            explanation,
            root_check,
            memory_log,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EqsatConfBuilder {
    pub iter_limit: Option<usize>,
    pub node_limit: Option<usize>,
    pub time_limit: Option<Duration>,
    pub explanation: bool,
    pub root_check: bool,
    pub memory_log: bool,
}

impl EqsatConfBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            iter_limit: None,
            node_limit: None,
            time_limit: None,
            explanation: false,
            root_check: false,
            memory_log: false,
        }
    }

    #[must_use]
    pub fn iter_limit(mut self, iter: usize) -> Self {
        self.iter_limit = Some(iter);
        self
    }

    #[must_use]
    pub fn node_limit(mut self, nodes: usize) -> Self {
        self.node_limit = Some(nodes);
        self
    }

    #[must_use]
    pub fn time_limit(mut self, time: Duration) -> Self {
        self.time_limit = Some(time);
        self
    }

    #[must_use]
    pub fn without_time_limit(mut self) -> Self {
        self.time_limit = None;
        self
    }

    #[must_use]
    pub fn with_explanation(mut self) -> Self {
        self.explanation = true;
        self
    }

    #[must_use]
    pub fn with_root_check(mut self) -> Self {
        self.root_check = true;
        self
    }

    #[must_use]
    pub fn with_memory_log(mut self) -> Self {
        self.root_check = true;
        self
    }

    #[must_use]
    pub fn build(self) -> EqsatConf {
        EqsatConf {
            iter_limit: self.iter_limit,
            node_limit: self.node_limit,
            time_limit: self.time_limit,
            explanation: self.explanation,
            root_check: self.root_check,
            memory_log: self.memory_log,
        }
    }
}

impl Default for EqsatConfBuilder {
    fn default() -> Self {
        Self::new()
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
    let mut runner = Runner::default();
    if let Some(iter_limit) = conf.iter_limit {
        info!("Setting iteration limit to {iter_limit}");
        runner = runner.with_iter_limit(iter_limit);
    };
    if let Some(node_limit) = conf.node_limit {
        info!("Setting node limit to {node_limit}");
        runner = runner.with_node_limit(node_limit);
    };
    if let Some(time_limit) = conf.time_limit {
        info!("Setting time limit to {time_limit:?}");
        runner = runner.with_time_limit(time_limit);
    };

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
