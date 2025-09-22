use egg::{Analysis, Language, RecExpr, Runner};
use hashbrown::HashSet;
use log::{info, warn};

pub const fn root_check_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String> + 'static
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

pub const fn goals_check_hook<L, N>(
    goal: RecExpr<L>,
) -> impl Fn(&mut Runner<L, N>) -> Result<(), String> + 'static
where
    L: Language + std::fmt::Display + 'static,
    N: Analysis<L> + Default,
{
    move |r: &mut Runner<L, N>| {
        if let Some(ids) = r.egraph.lookup_expr_ids(&goal) {
            if ids.iter().any(|id| r.roots.contains(id)) {
                return Err("Goal found".into());
            }
        }
        Ok(())
    }
}

#[expect(clippy::cast_precision_loss)]
pub fn memory_log_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String> + 'static
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
