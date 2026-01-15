use std::{fmt::Display, fs};

use egg::{Analysis, Language, RecExpr, Runner};
use hashbrown::{HashMap, HashSet};
use log::{info, warn};

pub const fn root_check_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String>
where
    L: Language,
    N: Analysis<L>,
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

pub const fn targe_hook<L, N>(
    target: RecExpr<L>,
) -> impl Fn(&mut Runner<L, N>) -> Result<(), String>
where
    L: Language,
    N: Analysis<L>,
{
    move |r: &mut Runner<L, N>| {
        if let Some(id) = r.egraph.lookup_expr(&target)
            && r.roots.contains(&r.egraph.find(id))
        {
            return Err("Target found".into());
        }
        Ok(())
    }
}

#[expect(clippy::missing_errors_doc)]
pub fn printer_hook<L, N>(r: &mut Runner<L, N>) -> Result<(), String>
where
    L: Language,
    N: Analysis<L>,
{
    let Some(this_iter) = r.iterations.last() else {
        return Ok(());
    };
    println!("ITERATION {}:\n", r.iterations.len());
    println!("Nodes: {}\n", r.egraph.nodes().len());
    println!("EClasses: {}\n", r.egraph.number_of_classes());
    println!("Rules:");
    for rule in &this_iter.applied {
        let name = &rule.0;
        let n = &rule.1;
        println!("{name}: {n}");
    }
    println!("---");
    Ok(())
}

#[expect(clippy::cast_precision_loss, clippy::missing_panics_doc)]
pub fn memory_log_hook<L, N>() -> impl Fn(&mut Runner<L, N>) -> Result<(), String>
where
    L: Language,
    N: Analysis<L>,
{
    let contents = fs::read_to_string("/proc/meminfo").expect("Could not read /proc/meminfo");
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

#[expect(clippy::missing_errors_doc)]
pub fn node_detail_hook<L, N>(r: &mut Runner<L, N>) -> Result<(), String>
where
    L: Language + Display,
    N: Analysis<L>,
{
    println!("-------\nNode Distribution:");
    let counts = r.egraph.nodes().iter().fold(HashMap::new(), |mut acc, n| {
        let node_name = n.to_string();
        acc.entry(node_name)
            .and_modify(|x| {
                *x += 1;
            })
            .or_insert(1);
        acc
    });
    for (k, v) in counts {
        println!("{k}: {v}");
    }
    Ok(())
}
