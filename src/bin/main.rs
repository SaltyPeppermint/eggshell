use core::panic;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::Local;
use clap::Parser;
use egg::{
    Analysis, AstSize, CostFunction, EGraph, FromOp, Id, Language, RecExpr, Rewrite,
    SimpleScheduler, StopReason,
};
use eggshell::explanation::{self, ExplanationData};
use hashbrown::HashSet;
use log::{debug, info};
use num::BigUint;
use rand::SeedableRng;

use eggshell::cli::{Cli, RewriteSystemName, SampleStrategy};
use eggshell::eqsat::{self, EqsatConf, EqsatResult};
use eggshell::io::reader;
use eggshell::io::structs::Entry;
use eggshell::rewrite_system::{Halide, RewriteSystem, Rise};
use eggshell::sampling::sampler::{
    CostWeighted, CountWeightedGreedy, CountWeightedUniformly, Sampler,
};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const BATCH_SIZE: usize = 1000;

fn main() {
    env_logger::init();
    let start_time = Local::now();
    let cli = Cli::parse();
    let eqsat_conf = EqsatConf::builder()
        .explanation(true)
        .maybe_memory_limit(cli.memory_limit())
        .maybe_iter_limit(cli.iter_limit())
        .build();
    let uuid = Uuid::new_v4();

    let folder: PathBuf = format!(
        "data/generated_samples/{}/{}-{}-{}",
        cli.rewrite_system(),
        cli.file().file_stem().unwrap().to_str().unwrap(),
        start_time.format("%y-%m-%d"),
        uuid
    )
    .into();
    fs::create_dir_all(&folder).unwrap();

    let exprs = match cli.file().extension().unwrap().to_str().unwrap() {
        "csv" => reader::read_exprs_csv(cli.file()),
        "json" => reader::read_exprs_json(cli.file()),
        extension => panic!("Unknown file extension {}", extension),
    };

    match cli.rewrite_system() {
        RewriteSystemName::Halide => {
            run::<Halide>(
                exprs,
                eqsat_conf,
                folder,
                start_time.timestamp(),
                &cli,
                uuid,
            );
        }
        RewriteSystemName::Rise => {
            run::<Rise>(
                exprs,
                eqsat_conf,
                folder,
                start_time.timestamp(),
                &cli,
                uuid,
            );
        }
    }

    let runtime = Local::now() - start_time;
    info!(
        "Runtime: {:0>2}:{:0>2}:{:0>2}",
        runtime.num_hours(),
        runtime.num_minutes() % 60,
        runtime.num_seconds() % 60
    );

    info!("EXPR {} DONE!", cli.expr_id());
}

fn run<R: RewriteSystem>(
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    experiment_folder: PathBuf,
    timestamp: i64,
    cli: &Cli,
    uuid: Uuid,
) {
    let rules = R::full_rules();
    let entry = exprs
        .into_iter()
        .nth(cli.expr_id())
        .expect("Must be in the file!");
    info!("Starting work on expr {}: {}...", cli.expr_id(), entry.expr);
    let term_folder = experiment_folder.join(cli.expr_id().to_string());
    fs::create_dir_all(&term_folder).unwrap();
    info!(
        "Will write to folder: {}/{}",
        experiment_folder.to_string_lossy(),
        cli.expr_id()
    );

    info!("Starting Eqsat...");
    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let midpoint_eqsats = eqsat(&start_expr, &eqsat_conf, rules.as_slice());
    let last_midpoint_eqsat = &midpoint_eqsats[midpoint_eqsats.len() - 1];
    let second_to_last_midpoint_egraph = &midpoint_eqsats[midpoint_eqsats.len() - 2].egraph();

    info!("Finished Eqsat!\nStarting sampling...");
    let midpoint_samples = sample_egraph(cli, &start_expr, last_midpoint_eqsat)
        .into_iter()
        .filter(|sample| second_to_last_midpoint_egraph.lookup_expr(sample).is_none())
        .collect::<Vec<_>>();
    info!(
        "Took {} unique samples while aiming for {}.",
        midpoint_samples.len(),
        cli.eclass_samples() * midpoint_eqsats.len()
    );

    let expl_counter = AtomicUsize::new(0);
    for (midpoint, midpoint_expl) in midpoint_samples.chunks(BATCH_SIZE).enumerate().fold(
        Vec::new(),
        |mut acc, (batch_id, sample_batch)| {
            info!("Starting explenations on batch {batch_id}...");
            let midpoints = par_expls(
                &start_expr,
                last_midpoint_eqsat.egraph(),
                &expl_counter,
                sample_batch,
            );
            info!("Finished work on batch {batch_id}!");
            acc.par_extend(midpoints);
            acc
        },
    ) {
        let second_eqsats = eqsat(&start_expr, &eqsat_conf, rules.as_slice());
        info!("Finished Eqsat!");

        let last_goal_eqsat = &second_eqsats[midpoint_eqsats.len() - 1];
        let second_to_last_goal_egraph = &second_eqsats[midpoint_eqsats.len() - 2].egraph();

        let goal_samples = sample_egraph(cli, &midpoint, last_goal_eqsat)
            .into_iter()
            .filter(|sample| {
                last_midpoint_eqsat.egraph().lookup_expr(sample).is_none()
                    && second_to_last_goal_egraph.lookup_expr(sample).is_none()
            })
            .collect::<Vec<_>>();

        let expl_counter = AtomicUsize::new(0);
        for (batch_id, sample_batch) in goal_samples.chunks(BATCH_SIZE).enumerate() {
            info!("Starting work on batch {batch_id}...");
            info!("Generating explanations...");
            let second_legs = par_expls(
                &midpoint,
                last_midpoint_eqsat.egraph(),
                &expl_counter,
                sample_batch,
            )
            .map(|(goal, goal_expl)| GoalSamples {
                sample: goal,
                explanation: goal_expl,
            })
            .collect();
            info!("Finished work on batch {batch_id}!");

            info!("Writing batch to disk at {batch_id}.json ...",);
            let data = DataEntry {
                start_expr: start_expr.clone(),
                sample_data: MidpointSamples {
                    midpoint: midpoint.clone(),
                    midpoint_expl: midpoint_expl.clone(),
                    second_legs,
                },
                metadata: MetaData {
                    batch_file: batch_id,
                    uuid: uuid.to_string(),
                    cli: cli.to_owned(),
                    timestamp,
                    eqsat_conf: eqsat_conf.clone(),
                    rules: rules.iter().map(|r| r.name.to_string()).collect(),
                },
            };

            let batch_file: PathBuf = term_folder
                .join(batch_id.to_string())
                .with_extension("json");

            let mut f = BufWriter::new(File::create(batch_file).unwrap());
            serde_json::to_writer(&mut f, &data).unwrap();
            f.flush().unwrap();
            drop(data);
            info!("Results for batch {batch_id} written to disk!");
        }
    }

    info!("Work on expr {} done!", cli.expr_id());
}
fn eqsat<L, N>(
    start_expr: &RecExpr<L>,
    eqsat_conf: &EqsatConf,
    rules: &[Rewrite<L, N>],
) -> Vec<EqsatResult<L, N>>
where
    L: Language + Display + Serialize + 'static,
    N: Analysis<L> + Clone + Serialize + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    let mut result = eqsat::eqsat(
        eqsat_conf.to_owned(),
        start_expr.into(),
        rules,
        None,
        &[],
        SimpleScheduler,
    );
    let mut eqsat_results = Vec::new();
    let mut iter_count = 0;

    loop {
        iter_count += 1;
        info!("Iteration {iter_count} stopped.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                eqsat_results.push(result.clone());

                result = eqsat::eqsat(
                    eqsat_conf.to_owned(),
                    result.into(),
                    rules,
                    None,
                    &[],
                    SimpleScheduler,
                );
            }
            _ => {
                info!("Limits reached after {} full iterations!", iter_count - 1);
                info!(
                    "Max Memory Consumption: {:?}",
                    result
                        .iterations()
                        .last()
                        .expect("Should be at least one")
                        .mem_usage
                );
                break;
            }
        }
    }
    eqsat_results
}

fn sample_egraph<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    eqsat_result: &EqsatResult<L, N>,
) -> Vec<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let eclass_id = eqsat_result.roots()[0];
    let mut rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());

    info!("Running sampling ...");
    let samples = sample(
        cli,
        start_expr,
        eqsat_result.egraph(),
        cli.eclass_samples(),
        eclass_id,
        &mut rng,
    );
    info!("Finished sampling!");
    samples.into_iter().collect()
}

/// Inner sample logic.
/// Samples guranteed to be unique.
fn sample<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    egraph: &EGraph<L, N>,
    n_samples: usize,
    eclass_id: Id,
    rng: &mut ChaCha12Rng,
) -> HashSet<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let parallelism = std::thread::available_parallelism().unwrap().into();
    let min_size = AstSize.cost_rec(start_expr);
    let max_size = min_size * 2;

    let exprs = match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            CountWeightedUniformly::<BigUint, _, _>::new(egraph, max_size)
                .sample_eclass(rng, n_samples, eclass_id, max_size, parallelism)
                .unwrap()
        }
        SampleStrategy::CountWeightedSizeAdjusted => {
            let sampler = CountWeightedUniformly::<BigUint, _, _>::new(egraph, max_size);
            let samples = (min_size..max_size)
                .into_par_iter()
                .enumerate()
                .map({
                    |(thread_idx, limit)| {
                        debug!("Sampling for size {limit}...");
                        let mut inner_rng = rng.clone();
                        inner_rng.set_stream((thread_idx + 1) as u64);
                        sampler
                            .sample_eclass(
                                &inner_rng,
                                n_samples / min_size,
                                eclass_id,
                                limit,
                                parallelism / 8,
                            )
                            .unwrap()
                    }
                })
                .reduce(HashSet::new, |mut a, b| {
                    a.extend(b);
                    a
                });

            info!("Sampled {} expressions!", samples.len());
            samples
        }
        SampleStrategy::CountWeightedGreedy => {
            CountWeightedGreedy::<BigUint, _, _>::new(egraph, max_size)
                .sample_eclass(rng, n_samples, eclass_id, start_expr.len(), parallelism)
                .unwrap()
        }
        SampleStrategy::CostWeighted => CostWeighted::new(egraph, AstSize)
            .sample_eclass(rng, n_samples, eclass_id, max_size, parallelism)
            .unwrap(),
    };
    exprs
}

fn par_expls<L, N>(
    start_expr: &RecExpr<L>,
    egraph: &EGraph<L, N>,
    expl_counter: &AtomicUsize,
    sample_batch: &[RecExpr<L>],
) -> impl ParallelIterator<Item = (RecExpr<L>, ExplanationData<L>)>
where
    L: Language + Display + FromOp + Send + Sync,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Send + Sync,
    N::Data: Clone + Serialize + Send + Sync,
{
    sample_batch
        .par_iter()
        .map_with(egraph.clone(), |thread_local_egraph, sample| {
            let explanation =
                explanation::explain_equivalence(thread_local_egraph, start_expr, sample);
            let c = expl_counter.fetch_add(1, Ordering::AcqRel);
            if (c + 1) % 100 == 0 {
                info!("Generated explanation {}...", c + 1);
            }
            (sample.to_owned(), explanation)
        })
}

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + FromOp + Display> {
    start_expr: RecExpr<L>,
    sample_data: MidpointSamples<L>,
    // baselines: Option<HashMap<usize, HashMap<usize, EqsatStats>>>,
    metadata: MetaData,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData {
    batch_file: usize,
    uuid: String,
    cli: Cli,
    timestamp: i64,
    eqsat_conf: EqsatConf,
    rules: Vec<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct MidpointSamples<L: Language + FromOp + Display> {
    midpoint: RecExpr<L>,
    midpoint_expl: ExplanationData<L>,
    second_legs: Vec<GoalSamples<L>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct GoalSamples<L: Language + FromOp + Display> {
    sample: RecExpr<L>,
    explanation: ExplanationData<L>,
}
