use core::panic;
use std::fmt::{Debug, Display};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;

use chrono::{DateTime, Local, TimeDelta};
use clap::Parser;
use egg::{
    Analysis, AstSize, CostFunction, EGraph, FromOp, Id, Language, RecExpr, Rewrite,
    SimpleScheduler, StopReason,
};
use hashbrown::HashSet;
use log::{debug, info, warn};
use num::BigUint;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use eggshell::cli::{Cli, RewriteSystemName, SampleStrategy};
use eggshell::eqsat::{self, EqsatConf, EqsatResult};
use eggshell::explanation::{self, ExplanationData};
use eggshell::io::{reader, structs::Entry};
use eggshell::rewrite_system::{Halide, RewriteSystem, Rise};
use eggshell::sampling::sampler::{CostWeighted, CountUniformly, Greedy, Sampler};

fn main() {
    env_logger::init();
    let start_time = Local::now();
    let cli = Cli::parse();
    let eqsat_conf = EqsatConf::builder()
        .explanation(cli.explanation())
        .maybe_node_limit(cli.node_limit())
        .iter_limit(1) // Important to capture the egraph after every iteration!
        .build();

    let folder: PathBuf = format!(
        "data/generated_samples/{}/{}-{}",
        cli.rewrite_system(),
        cli.file().file_stem().unwrap().to_str().unwrap(),
        start_time.to_rfc3339()
    )
    .into();
    fs::create_dir_all(&folder).unwrap();

    let exprs = match cli.file().extension().unwrap().to_str().unwrap() {
        "csv" => reader::read_exprs_csv(cli.file()),
        "json" => reader::read_exprs_json(cli.file()),
        extension => panic!("Unknown file extension {}", extension),
    };

    let entry = exprs
        .into_iter()
        .nth(cli.expr_id())
        .expect("Entry not found!");
    println!("Starting work on expr {}: {}...", cli.expr_id(), entry.expr);

    let term_folder = folder.join(cli.expr_id().to_string());
    fs::create_dir_all(&term_folder).unwrap();
    println!("Will write to folder: {}", term_folder.to_string_lossy());

    match cli.rewrite_system() {
        RewriteSystemName::Halide => {
            run::<Halide>(entry, eqsat_conf, term_folder, start_time, &cli);
        }
        RewriteSystemName::Rise => {
            run::<Rise>(entry, eqsat_conf, term_folder, start_time, &cli);
        }
    }

    print_delta(Local::now() - start_time);
    println!("Work on expression {} done!", cli.expr_id());
}

fn run<R: RewriteSystem>(
    entry: Entry,
    eqsat_conf: EqsatConf,
    term_folder: PathBuf,
    start_time: DateTime<Local>,
    cli: &Cli,
) {
    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let rules = R::full_rules();
    let rule_names = rules.iter().map(|r| r.name.to_string()).collect::<Vec<_>>();

    let (midpoint_eqsat, midpoints, midpoint_iterations) =
        midpoints(&eqsat_conf, cli, &rules, &start_expr);

    let samples_per_midpoint = midpoints
        .into_par_iter()
        .enumerate()
        .map(|(midpoint_id, midpoint)| {
            info!("Now working on the goals for midpoint {midpoint_id}:");
            let (goal_eqsat, goal_samples, goals_iterations) =
                goals(&eqsat_conf, cli, &rules, &midpoint, midpoint_eqsat.egraph());
            if goal_samples.len() == 0 {
                warn!("No goal samples taken for midpoint {midpoint_id}");
            }
            info!("Took {} unique goal samples.", goal_samples.len());

            let midpoint_expl = cli
                .explanation()
                .then(|| explain(&start_expr, &midpoint, midpoint_eqsat.egraph()));
            let midpoint_sample = SampleWithExpl::new(&midpoint, midpoint_expl);

            goal_samples
                .chunks(cli.batch_size())
                .enumerate()
                .map(|(batch_id, sample_batch)| {
                    info!("Working on goal batch {batch_id}...");

                    let goals = sample_batch
                        .into_iter()
                        .map(|goal| {
                            let goal_expl = cli
                                .explanation()
                                .then(|| explain(&midpoint, &goal, goal_eqsat.egraph()));
                            SampleWithExpl::new(goal, goal_expl)
                        })
                        .collect::<Vec<_>>();

                    let metadata = MetaData::new(cli, &start_time, &eqsat_conf, &rule_names);
                    let midpoint = Midpoint::new(&midpoint_sample, goals_iterations, goals);
                    let data =
                        DataEntry::new(&start_expr, midpoint_iterations, midpoint, &metadata);

                    save_batch(&term_folder, midpoint_id, batch_id, &data);
                    info!("Finished work on goal batch {batch_id} of midpoint {midpoint_id}!");
                    data.midpoint.goals.len()
                })
                .sum::<usize>()
        })
        .collect::<Vec<_>>();

    let total_midpoints = samples_per_midpoint.len();
    let total_samples = samples_per_midpoint.iter().sum::<usize>();
    let average_samples = total_samples / samples_per_midpoint.len();
    let midpoints_no_samples = samples_per_midpoint.iter().filter(|x| **x == 0).count();

    println!("Total Midpoints: {total_midpoints}");
    println!("Total Samples: {total_samples}");
    println!("Average Samples per Midpoint: {average_samples}");
    println!("Midpoints with no samples: {midpoints_no_samples}");
}

fn explain<L, N>(from: &RecExpr<L>, to: &RecExpr<L>, egraph: &EGraph<L, N>) -> ExplanationData<L>
where
    L: Language + Display + FromOp + Clone,
    N: Analysis<L> + Clone + ToOwned,
    N::Data: Clone,
{
    let mut temp_egraph = egraph.to_owned();
    explanation::explain_equivalence(&mut temp_egraph, from, to)
}

fn midpoints<L, N>(
    eqsat_conf: &EqsatConf,
    cli: &Cli,
    rules: &Vec<Rewrite<L, N>>,
    start_expr: &RecExpr<L>,
) -> (EqsatResult<L, N>, Vec<RecExpr<L>>, usize)
where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    info!("Starting Midpoint Eqsat...");
    let (eqsat, penultimate_eqsat, iterations) =
        run_eqsat(start_expr, eqsat_conf, rules.as_slice(), cli.iter_limit());
    info!("Finished Midpoint Eqsat!");

    info!("Starting sampling...");
    let raw_samples = sample_egraph(cli, start_expr, eqsat.egraph(), eqsat.roots()[0]);
    info!("Finished sampling, now filtering...");
    let filtered_samples: Vec<_> = if let Some(e) = penultimate_eqsat {
        raw_samples
            .filter(|sample| e.egraph().lookup_expr(sample).is_none())
            .collect()
    } else {
        warn!("Only one iteration was possible in the midpoint eqsat");
        raw_samples.collect()
    };
    info!("Took {} unique midpoint samples.", filtered_samples.len());

    (eqsat, filtered_samples, iterations)
}

fn goals<L, N>(
    eqsat_conf: &EqsatConf,
    cli: &Cli,
    rules: &[Rewrite<L, N>],
    midpoint: &RecExpr<L>,
    midpoint_egraph: &EGraph<L, N>,
) -> (EqsatResult<L, N>, Vec<RecExpr<L>>, usize)
where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    info!("Starting Goal Eqsat...");
    let (last_goal_eqsat, penultimate_eqsat, iterations) =
        run_eqsat(midpoint, eqsat_conf, rules, cli.iter_limit());
    info!("Finished Goal Eqsat!");

    let raw_samples = sample_egraph(
        cli,
        midpoint,
        last_goal_eqsat.egraph(),
        last_goal_eqsat.roots()[0],
    );
    info!("{} raw samples!", raw_samples.len());
    let filtered_samples: Vec<_> = if let Some(penultimate_egraph) = penultimate_eqsat {
        raw_samples
            .filter(|sample| {
                let not_in_midpoint = midpoint_egraph.lookup_expr(sample).is_none();
                let not_in_penultimate = penultimate_egraph.egraph().lookup_expr(sample).is_none();
                not_in_midpoint && not_in_penultimate
            })
            .collect()
    } else {
        warn!("Only one iteration was possible in this goal eqsat");
        raw_samples
            .filter(|sample| midpoint_egraph.lookup_expr(sample).is_none())
            .collect()
    };

    (last_goal_eqsat, filtered_samples, iterations)
}

fn run_eqsat<L, N>(
    start_expr: &RecExpr<L>,
    eqsat_conf: &EqsatConf,
    rules: &[Rewrite<L, N>],
    iter_limit: Option<usize>,
) -> (EqsatResult<L, N>, Option<EqsatResult<L, N>>, usize)
where
    L: Language + Display + Serialize + 'static,
    N: Analysis<L> + Clone + Serialize + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    let mut start_material = start_expr.into();
    let mut last_eqsat = None;
    let mut penultimate_eqsat = None;
    let mut iter_count = 0;

    loop {
        let result = eqsat::eqsat(
            eqsat_conf.to_owned(),
            start_material,
            rules,
            None,
            SimpleScheduler,
        );
        if let StopReason::IterationLimit(_) = result.report().stop_reason
            && iter_limit.map(|limit| iter_count < limit).unwrap_or(true)
        {
            iter_count += 1;
            penultimate_eqsat = last_eqsat;
            last_eqsat = Some(result.clone());
            start_material = result.into()
        } else {
            info!("Limits reached after {iter_count} full iterations!");
            break (
                last_eqsat.expect("At least one iteration eqsat has to be run"),
                penultimate_eqsat,
                iter_count,
            );
        }
    }
}

fn sample_egraph<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    egraph: &EGraph<L, N>,
    eclass_id: Id,
) -> impl ExactSizeIterator<Item = RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());
    let n_samples = cli.eclass_samples();
    let parallelism = cli
        .sample_parallelism()
        .unwrap_or(usize::from(thread::available_parallelism().unwrap()));

    let min_size = AstSize.cost_rec(start_expr);
    let max_size = min_size * 2;
    info!("Running sampling ...");
    let samples = match &cli.strategy() {
        SampleStrategy::CountUniformly => CountUniformly::<BigUint, _, _>::new(egraph, max_size)
            .sample_eclass(&rng, n_samples, eclass_id, max_size, parallelism)
            .unwrap(),
        SampleStrategy::CountSizeRange => {
            let sampler = CountUniformly::<BigUint, _, _>::new(egraph, max_size);
            (min_size..max_size)
                .into_par_iter()
                .enumerate()
                .map({
                    |(thread_idx, limit)| {
                        debug!("Sampling for size {limit}...");
                        let mut inner_rng = rng.clone();
                        inner_rng.set_stream((thread_idx + 1) as u64);
                        sampler
                            .sample_eclass(&inner_rng, n_samples, eclass_id, limit, parallelism)
                            .unwrap()
                    }
                })
                .reduce(HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                })
        }
        SampleStrategy::Greedy => Greedy::new(egraph)
            .sample_eclass(&rng, n_samples, eclass_id, max_size, parallelism)
            .unwrap(),
        SampleStrategy::CostWeighted => CostWeighted::new(egraph, AstSize)
            .sample_eclass(&rng, n_samples, eclass_id, max_size, parallelism)
            .unwrap(),
    };
    info!("Finished sampling!");
    samples.into_iter()
}

fn save_batch<L: Language + FromOp + Display + Serialize>(
    term_folder: &PathBuf,
    midpoint_id: usize,
    batch_id: usize,
    data: &DataEntry<L>,
) {
    let batch_file = term_folder
        .join(format!("{midpoint_id}-{batch_id}"))
        .with_extension("json");
    info!("Writing batch to {}...", batch_file.to_string_lossy());

    let mut f = BufWriter::new(File::create(batch_file).unwrap());
    serde_json::to_writer(&mut f, data).unwrap();
    f.flush().unwrap();
    info!("Results for batch written to disk!");
}

fn print_delta(delta: TimeDelta) {
    info!(
        "Runtime: {:0>2}:{:0>2}:{:0>2}",
        delta.num_hours(),
        delta.num_minutes() % 60,
        delta.num_seconds() % 60
    );
}

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<'a, L: Language + FromOp + Display> {
    start_expr: &'a RecExpr<L>,
    iterations: usize,
    midpoint: Midpoint<'a, L>,
    metadata: &'a MetaData<'a>,
}

impl<'a, L: Language + FromOp + Display> DataEntry<'a, L> {
    pub fn new(
        start_expr: &'a RecExpr<L>,
        iterations: usize,
        midpoint: Midpoint<'a, L>,
        metadata: &'a MetaData,
    ) -> Self {
        Self {
            start_expr,
            iterations,
            midpoint,
            metadata,
        }
    }
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData<'a> {
    cli: &'a Cli,
    start_time: String,
    eqsat_conf: &'a EqsatConf,
    rule_names: &'a [String],
}

impl<'a> MetaData<'a> {
    pub fn new(
        cli: &'a Cli,
        start_time: &DateTime<Local>,
        eqsat_conf: &'a EqsatConf,
        rules: &'a [String],
    ) -> Self {
        Self {
            cli,
            start_time: start_time.to_rfc3339(),
            eqsat_conf,
            rule_names: rules,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Midpoint<'a, L: Language + FromOp + Display> {
    midpoint: &'a SampleWithExpl<'a, L>,
    iterations: usize,
    goals: Vec<SampleWithExpl<'a, L>>,
}

impl<'a, L: Language + FromOp + Display> Midpoint<'a, L> {
    pub fn new(
        midpoint: &'a SampleWithExpl<L>,
        iterations: usize,
        goals: Vec<SampleWithExpl<'a, L>>,
    ) -> Self {
        Self {
            midpoint,
            iterations,
            goals,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct SampleWithExpl<'a, L: Language + FromOp + Display> {
    expression: &'a RecExpr<L>,
    explanation: Option<ExplanationData<L>>,
}

impl<'a, L: Language + FromOp + Display> SampleWithExpl<'a, L> {
    pub fn new(expression: &'a RecExpr<L>, explanation: Option<ExplanationData<L>>) -> Self {
        Self {
            expression,
            explanation,
        }
    }
}
