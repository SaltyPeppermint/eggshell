use core::panic;
use std::fmt::{Debug, Display};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
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
use serde::{Deserialize, Serialize};

use eggshell::cli::{Cli, RewriteSystemName, SampleStrategy};
use eggshell::eqsat::{self, EqsatConf, EqsatResult};
use eggshell::explanation::{self, ExplanationData};
use eggshell::io::{reader, structs::Entry};
use eggshell::rewrite_system::{Halide, RewriteSystem, Rise};
use eggshell::sampling::sampler::{
    CostWeighted, CountWeightedGreedy, CountWeightedUniformly, Sampler,
};

fn main() {
    env_logger::init();
    let start_time = Local::now();
    let cli = Cli::parse();
    let eqsat_conf = EqsatConf::builder()
        .explanation(true)
        .maybe_memory_limit(cli.memory_limit())
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

    match cli.rewrite_system() {
        RewriteSystemName::Halide => {
            run::<Halide>(exprs, eqsat_conf, folder, start_time, &cli);
        }
        RewriteSystemName::Rise => {
            run::<Rise>(exprs, eqsat_conf, folder, start_time, &cli);
        }
    }

    print_delta(Local::now() - start_time);
    info!("EXPR {} DONE!", cli.expr_id());
}

fn run<R: RewriteSystem>(
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    experiment_folder: PathBuf,
    start_time: DateTime<Local>,
    cli: &Cli,
) {
    let entry = exprs
        .into_iter()
        .nth(cli.expr_id())
        .expect("Entry not found!");
    info!("Starting work on expr {}: {}...", cli.expr_id(), entry.expr);
    let term_folder = experiment_folder.join(cli.expr_id().to_string());
    fs::create_dir_all(&term_folder).unwrap();
    info!("Will write to folder: {}", term_folder.to_string_lossy());

    let rules = R::full_rules();

    info!("Starting Midpoint Eqsat...");
    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let (last_midpoint_eqsat, second_to_last_midpoint_eqsat) =
        run_eqsat(&start_expr, &eqsat_conf, rules.as_slice());
    info!("Finished Midpoint Eqsat!");

    info!("Starting sampling...");
    let raw_midpoint_samples = sample_egraph(
        cli,
        &start_expr,
        last_midpoint_eqsat.egraph(),
        last_midpoint_eqsat.roots()[0],
    );
    info!("Finished sampling, now filtering...");
    let midpoint_samples: Vec<_> = if let Some(e) = second_to_last_midpoint_eqsat {
        raw_midpoint_samples
            .filter(|sample| e.egraph().lookup_expr(sample).is_none())
            .collect()
    } else {
        warn!("Only one iteration was possible in the midpoint eqsat");
        raw_midpoint_samples.collect()
    };
    info!("Took {} unique midpoint samples.", midpoint_samples.len());

    let midpoint_expl_counter = AtomicUsize::new(0);
    let midpoints_with_expls = midpoint_samples.chunks(cli.batch_size()).enumerate().fold(
        Vec::new(),
        |mut acc, (batch_id, sample_batch)| {
            info!("Starting explenations on midpoint batch {batch_id}...");
            let midpoints = par_expls(
                &start_expr,
                last_midpoint_eqsat.egraph(),
                &midpoint_expl_counter,
                sample_batch,
            );
            info!("Finished work on midpoint batch {batch_id}!");
            acc.par_extend(midpoints);
            acc
        },
    );

    let goal_expl_counter = AtomicUsize::new(0);
    for (midpoint_id, (midpoint, midpoint_expl)) in midpoints_with_expls.into_iter().enumerate() {
        info!("Starting Goal Eqsat {midpoint_id}...");
        let (last_goal_eqsat, second_to_last_goal_eqsat) =
            run_eqsat(&start_expr, &eqsat_conf, rules.as_slice());
        info!("Finished Goal Eqsat!");

        let raw_goal_samples = sample_egraph(
            cli,
            &midpoint,
            last_goal_eqsat.egraph(),
            last_goal_eqsat.roots()[0],
        );
        let goal_samples: Vec<_> = if let Some(e) = second_to_last_goal_eqsat {
            raw_goal_samples
                .filter(|sample| {
                    last_midpoint_eqsat.egraph().lookup_expr(sample).is_none()
                        && e.egraph().lookup_expr(sample).is_none()
                })
                .collect()
        } else {
            warn!("Only one iteration was possible in the goal eqsat");
            raw_goal_samples
                .filter(|sample| last_midpoint_eqsat.egraph().lookup_expr(sample).is_none())
                .collect()
        };
        info!(
            "Took {} unique goal samples for midpoint {}.",
            goal_samples.len(),
            midpoint_id
        );

        for (batch_id, sample_batch) in goal_samples.chunks(cli.batch_size()).enumerate() {
            info!("Generating explanations for goal batch {midpoint_id}-{batch_id}...");
            let goal = par_expls(
                &midpoint,
                last_midpoint_eqsat.egraph(),
                &goal_expl_counter,
                sample_batch,
            )
            .map(|(expr, expl)| Goal::new(expr, expl))
            .collect();
            info!("Finished work on goal batch {midpoint_id}-{batch_id}!");

            let data = DataEntry {
                start_expr: start_expr.clone(),
                midpoint: Midpoint::new(midpoint.clone(), midpoint_expl.clone(), goal),
                metadata: MetaData::new(
                    cli.to_owned(),
                    start_time.to_rfc3339(),
                    eqsat_conf.clone(),
                    rules.iter().map(|r| r.name.to_string()).collect(),
                ),
            };

            let batch_file = term_folder
                .join(format!("{midpoint_id}-{batch_id}"))
                .with_extension("json");
            info!("Writing batch to {}...", batch_file.to_string_lossy());

            let mut f = BufWriter::new(File::create(batch_file).unwrap());
            serde_json::to_writer(&mut f, &data).unwrap();
            f.flush().unwrap();
            drop(data);
            info!("Results for batch {midpoint_id}-{batch_id} written to disk!");
        }
    }

    info!("Work on expr {} done!", cli.expr_id());
}
fn run_eqsat<L, N>(
    start_expr: &RecExpr<L>,
    eqsat_conf: &EqsatConf,
    rules: &[Rewrite<L, N>],
) -> (EqsatResult<L, N>, Option<EqsatResult<L, N>>)
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
    let mut second_to_last_eqsat = None;
    let mut iter_count = 0;

    let last_eqsat = loop {
        iter_count += 1;
        info!("Iteration {iter_count} stopped.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                second_to_last_eqsat = Some(result.clone());
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
                break result;
            }
        }
    };
    (last_eqsat, second_to_last_eqsat)
}

fn sample_egraph<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    egraph: &EGraph<L, N>,
    eclass_id: Id,
) -> impl Iterator<Item = RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());
    let n_samples = cli.eclass_samples();
    let parallelism = usize::from(thread::available_parallelism().unwrap()) / 8;

    let min_size = AstSize.cost_rec(start_expr);
    let max_size = min_size * 2;
    info!("Running sampling ...");
    let samples = match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            CountWeightedUniformly::<BigUint, _, _>::new(egraph, max_size)
                .sample_eclass(&rng, n_samples, eclass_id, max_size, parallelism)
                .unwrap()
        }
        SampleStrategy::CountWeightedSizeAdjusted => (min_size..max_size)
            .into_par_iter()
            .enumerate()
            .map({
                |(thread_idx, limit)| {
                    debug!("Sampling for size {limit}...");
                    let mut inner_rng = rng.clone();
                    inner_rng.set_stream((thread_idx + 1) as u64);
                    CountWeightedUniformly::<BigUint, _, _>::new(egraph, max_size)
                        .sample_eclass(&inner_rng, n_samples, eclass_id, limit, parallelism)
                        .unwrap()
                }
            })
            .reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                a
            }),
        SampleStrategy::CountWeightedGreedy => {
            CountWeightedGreedy::<BigUint, _, _>::new(egraph, max_size)
                .sample_eclass(&rng, n_samples, eclass_id, start_expr.len(), parallelism)
                .unwrap()
        }
        SampleStrategy::CostWeighted => CostWeighted::new(egraph, AstSize)
            .sample_eclass(&rng, n_samples, eclass_id, max_size, parallelism)
            .unwrap(),
    };
    info!("Finished sampling!");
    samples.into_iter()
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

fn print_delta(delta: TimeDelta) {
    info!(
        "Runtime: {:0>2}:{:0>2}:{:0>2}",
        delta.num_hours(),
        delta.num_minutes() % 60,
        delta.num_seconds() % 60
    );
}

#[derive(Serialize, Clone, Debug)]
pub struct DataEntry<L: Language + FromOp + Display> {
    start_expr: RecExpr<L>,
    midpoint: Midpoint<L>,
    metadata: MetaData,
}

impl<L: Language + FromOp + Display> DataEntry<L> {
    pub fn new(start_expr: RecExpr<L>, midpoint: Midpoint<L>, metadata: MetaData) -> Self {
        Self {
            start_expr,
            midpoint,
            metadata,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData {
    cli: Cli,
    start_time: String,
    eqsat_conf: EqsatConf,
    rules: Vec<String>,
}

impl MetaData {
    pub fn new(cli: Cli, start_time: String, eqsat_conf: EqsatConf, rules: Vec<String>) -> Self {
        Self {
            cli,
            start_time,
            eqsat_conf,
            rules,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Midpoint<L: Language + FromOp + Display> {
    expr: RecExpr<L>,
    expl: ExplanationData<L>,
    second_legs: Vec<Goal<L>>,
}

impl<L: Language + FromOp + Display> Midpoint<L> {
    pub fn new(expr: RecExpr<L>, expl: ExplanationData<L>, second_legs: Vec<Goal<L>>) -> Self {
        Self {
            expr,
            expl,
            second_legs,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct Goal<L: Language + FromOp + Display> {
    sample: RecExpr<L>,
    explanation: ExplanationData<L>,
}

impl<L: Language + FromOp + Display> Goal<L> {
    pub fn new(sample: RecExpr<L>, explanation: ExplanationData<L>) -> Self {
        Self {
            sample,
            explanation,
        }
    }
}
