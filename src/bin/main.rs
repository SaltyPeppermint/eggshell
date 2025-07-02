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
    Analysis, AstSize, CostFunction, EGraph, FromOp, Language, RecExpr, Rewrite, StopReason,
};
use eggshell::explanation::{self, ExplanationData};
use hashbrown::HashSet;
use log::{debug, info};
use num::BigUint;
use rand::SeedableRng;

use eggshell::cli::{Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult, StartMaterial};
use eggshell::io::reader;
use eggshell::io::structs::Entry;
use eggshell::rewrite_system::{Halide, RewriteSystem, Rise};
use eggshell::sampling::sampler::{
    CostWeighted, CountWeightedGreedy, CountWeightedUniformly, Sampler,
};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn main() {
    env_logger::init();
    let start_time = Local::now();

    let cli = Cli::parse();

    let eqsat_conf = (&cli).into();

    let folder: PathBuf = format!(
        "data/generated_samples/{}/{}-{}-{}",
        cli.trs(),
        cli.file().file_stem().unwrap().to_str().unwrap(),
        start_time.format("%Y-%m-%d"),
        cli.uuid()
    )
    .into();
    fs::create_dir_all(&folder).unwrap();

    let exprs = match cli.file().extension().unwrap().to_str().unwrap() {
        "csv" => reader::read_exprs_csv(cli.file()),
        "json" => reader::read_exprs_json(cli.file()),
        extension => panic!("Unknown file extension {}", extension),
    };

    match cli.trs() {
        TrsName::Halide => {
            run::<Halide>(
                exprs,
                eqsat_conf,
                folder,
                start_time.timestamp(),
                Halide::full_rules(),
                &cli,
            );
        }
        TrsName::Rise => {
            run::<Rise>(
                exprs,
                eqsat_conf,
                folder,
                start_time.timestamp(),
                Rise::full_rules(),
                &cli,
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
    rules: Vec<Rewrite<R::Language, R::Analysis>>,
    cli: &Cli,
) {
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
    let mut eqsat_results = eqsat(&start_expr, &eqsat_conf, rules.as_slice());
    info!("Finished Eqsat!");

    info!("Starting sampling...");
    let samples = sample_generations(cli, &start_expr, &eqsat_results);
    info!(
        "Took {} unique samples while aiming for {}.",
        samples.len(),
        cli.eclass_samples() * eqsat_results.len()
    );

    // let max_generation = eqsat_results.len();
    let samples_with_gen = with_generations(samples, &mut eqsat_results);
    let last_egraph = eqsat_results.last().unwrap().egraph().to_owned();
    drop(eqsat_results);

    run_batch(
        eqsat_conf,
        term_folder,
        timestamp,
        rules,
        cli,
        start_expr,
        &last_egraph,
        samples_with_gen,
    );
    info!("Work on expr {} done!", cli.expr_id());
}

#[expect(clippy::too_many_arguments)]
fn run_batch<L, N>(
    eqsat_conf: EqsatConf,
    term_folder: PathBuf,
    timestamp: i64,
    rules: Vec<Rewrite<L, N>>,
    cli: &Cli,
    start_expr: RecExpr<L>,
    last_egraph: &EGraph<L, N>,
    samples_with_gen: Vec<(RecExpr<L>, usize)>,
) where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize,
    L::Discriminant: Sync + Send,
    N: Analysis<L> + Clone + Debug + Send + Sync,
    N::Data: Serialize + Clone + Sync + Send,
{
    let expl_counter = AtomicUsize::new(0);
    const BATCH_SIZE: usize = 1000;

    info!("Working in batches of size {BATCH_SIZE}...");
    for (batch_id, sample_batch) in samples_with_gen.chunks(BATCH_SIZE).enumerate() {
        info!("Starting work on batch {batch_id}...");
        info!("Generating explanations...");
        let sample_data = sample_batch
            .par_iter()
            .map_with(
                last_egraph.clone(),
                |thread_local_eqsat_results, (sample, generation)| {
                    let explanation = cli.with_explanations().then(|| {
                        let expl = explanation::explain_equivalence(
                            thread_local_eqsat_results,
                            &start_expr,
                            sample,
                        );
                        let c = expl_counter.fetch_add(1, Ordering::AcqRel);
                        if (c + 1) % 100 == 0 {
                            info!("Generated explanation {}...", c + 1);
                        }
                        expl
                    });
                    SampleData {
                        sample: sample.to_owned(),
                        generation: *generation,
                        explanation,
                    }
                },
            )
            .collect::<Vec<_>>();
        info!("Finished generating explanations!");

        info!("Finished work on batch {batch_id}!");

        let batch_file: PathBuf = term_folder
            .join(batch_id.to_string())
            .with_extension("json");

        info!(
            "Writing batch to disk at {}...",
            batch_file.to_string_lossy()
        );

        let data = DataEntry {
            start_expr: start_expr.clone(),
            sample_data,
            // baselines,
            metadata: MetaData {
                uuid: cli.uuid().to_owned(),
                folder: batch_file.to_string_lossy().into(),
                cli: cli.to_owned(),
                timestamp,
                eqsat_conf: eqsat_conf.clone(),
                rules: rules.iter().map(|r| r.name.to_string()).collect(),
            },
        };

        let mut f = BufWriter::new(File::create(batch_file).unwrap());
        serde_json::to_writer(&mut f, &data).unwrap();
        f.flush().unwrap();
        drop(data);
        info!("Results for batch {batch_id} written to disk!");
    }
}

fn sample_generations<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    eqsat_results: &[EqsatResult<L, N>],
) -> Vec<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let mut rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());

    let samples: Vec<_> = eqsat_results
        .iter()
        .enumerate()
        .flat_map(|(eqsat_generation, eqsat_result)| {
            info!("Running sampling of generation {}...", eqsat_generation + 1);
            let s = sample(
                cli,
                start_expr,
                eqsat_result,
                cli.eclass_samples(),
                &mut rng,
            );
            info!("Finished sampling of generation {}!", eqsat_generation + 1);
            s
        })
        .collect();
    info!("Finished sampling!");
    samples
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
    let exprs = [start_expr];
    let mut eqsat =
        Eqsat::new(StartMaterial::RecExprs(&exprs), rules).with_conf(eqsat_conf.to_owned());
    let mut eqsat_results = Vec::new();
    let mut iter_count = 0;

    loop {
        let result = eqsat.run();
        iter_count += 1;
        info!("Iteration {iter_count} stopped.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                eqsat_results.push(result.clone());
                eqsat = Eqsat::new(result.into(), rules).with_conf(eqsat_conf.to_owned());
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

/// Inner sample logic.
/// Samples guranteed to be unique.
fn sample<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    eqsat: &EqsatResult<L, N>,
    n_samples: usize,
    rng: &mut ChaCha12Rng,
) -> HashSet<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let root_id = eqsat.roots()[0];
    let parallelism = std::thread::available_parallelism().unwrap().into();
    let min_size = AstSize.cost_rec(start_expr);
    let max_size = min_size * 2;

    match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            CountWeightedUniformly::<BigUint, _, _>::new(eqsat.egraph(), max_size)
                .sample_eclass(rng, n_samples, root_id, max_size, parallelism)
                .unwrap()
        }
        SampleStrategy::CountWeightedSizeAdjusted => {
            let sampler = CountWeightedUniformly::<BigUint, _, _>::new(eqsat.egraph(), max_size);
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
                                root_id,
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
            CountWeightedGreedy::<BigUint, _, _>::new(eqsat.egraph(), max_size)
                .sample_eclass(rng, n_samples, root_id, start_expr.len(), parallelism)
                .unwrap()
        }
        SampleStrategy::CostWeighted => CostWeighted::new(eqsat.egraph(), AstSize)
            .sample_eclass(rng, n_samples, root_id, max_size, parallelism)
            .unwrap(),
    }
}

fn with_generations<L, N>(
    samples: Vec<RecExpr<L>>,
    eqsat_results: &mut [EqsatResult<L, N>],
) -> Vec<(RecExpr<L>, usize)>
where
    L: Language + Display + FromOp,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    samples
        .into_iter()
        .map(|sample| {
            let generation = eqsat_results
                .iter_mut()
                .map(|r| r.egraph())
                .position(|egraph| egraph.lookup_expr(&sample).is_some())
                .expect("Must be in at least one of the egraphs")
                + 1;
            (sample, generation)
        })
        .collect::<Vec<_>>()
}

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + FromOp + Display> {
    start_expr: RecExpr<L>,
    sample_data: Vec<SampleData<L>>,
    // baselines: Option<HashMap<usize, HashMap<usize, EqsatStats>>>,
    metadata: MetaData,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData {
    uuid: String,
    folder: String,
    cli: Cli,
    timestamp: i64,
    eqsat_conf: EqsatConf,
    rules: Vec<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct SampleData<L: Language + FromOp + Display> {
    sample: RecExpr<L>,
    generation: usize,
    explanation: Option<ExplanationData<L>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct EqsatStats {
    stop_reason: StopReason,
    total_time: f64,
    total_nodes: usize,
    total_iters: usize,
}

impl<L, N> From<EqsatResult<L, N>> for EqsatStats
where
    L: Language + Display,
    N: Analysis<L> + Clone,
    N::Data: Serialize + Clone,
{
    fn from(result: EqsatResult<L, N>) -> Self {
        EqsatStats {
            stop_reason: result.report().stop_reason.clone(),
            total_time: result.report().total_time,
            total_nodes: result.report().egraph_nodes,
            total_iters: result.report().iterations,
        }
    }
}
