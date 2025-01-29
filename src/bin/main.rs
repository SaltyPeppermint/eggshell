use core::panic;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use chrono::Local;
use clap::Parser;
use egg::{Analysis, AstSize, CostFunction, FromOp, Language, RecExpr, Rewrite, StopReason};
use eggshell::explanation::{self, ExplanationData};
use hashbrown::HashSet;
use log::{debug, info};
use num::BigUint;
use rand::SeedableRng;

use eggshell::cli::{Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult, StartMaterial};
use eggshell::io::reader;
use eggshell::io::structs::Entry;
use eggshell::sampling::sampler::{
    CostWeighted, CountWeightedGreedy, CountWeightedUniformly, Sampler,
};
use eggshell::trs::{Halide, Rise, TermRewriteSystem};
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

fn run<R: TermRewriteSystem>(
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
    let generations = generations(&samples, &mut eqsat_results);

    let last_egraph = eqsat_results.last().unwrap().egraph().clone();
    drop(eqsat_results);
    let expl_counter = AtomicUsize::new(0);
    let batch_size = 1000;

    info!("Working in batches of size {batch_size}...");
    for (batch_id, sample_batch) in samples.chunks(1000).enumerate() {
        info!("Starting work on batch {}...", batch_id);
        info!("Generating explanations...");
        let sample_data = sample_batch
            .par_iter()
            .enumerate()
            .map_with(
                last_egraph.clone(),
                |thread_local_eqsat_results, (idx, sample)| {
                    let explanation = cli.with_explanations().then(|| {
                        let c = expl_counter.fetch_add(1, Ordering::AcqRel);
                        if c % 100 == 0 {
                            info!("Now generating explanations {}...", c);
                        }
                        explanation::explain_equivalence(
                            thread_local_eqsat_results,
                            &start_expr,
                            sample,
                        )
                    });
                    SampleData {
                        sample: sample.to_owned(),
                        generation: generations[idx + batch_id * batch_size],
                        explanation,
                    }
                },
            )
            .collect::<Vec<_>>();
        info!("Finished generating explanations!");

        info!("Finished work on batch {}!", batch_id);

        let batch_file: PathBuf = experiment_folder
            .join(batch_id.to_string())
            .with_extension("json");
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
        info!("Results for batch {} written to disk!", batch_id);
    }
    info!("Work on expr {} done!", cli.expr_id());

    // let pool = rayon::ThreadPoolBuilder::new()
    //     .num_threads(16) // Adjust if memory issues std::thread::available_parallelism::available_parallelism().unwrap().into()
    //     .build()
    //     .unwrap();
    // All explanations work on the last egraph
    // let explanations = pool.install(|| {
    //     samples
    //         .par_iter()
    //         .map_with(last_egraph, |thread_local_eqsat_results, sample| {
    //             cli.with_explanations().then(|| {
    //                 explanation::explain_equivalence(
    //                     thread_local_eqsat_results,
    //                     &start_expr,
    //                     sample,
    //                 )
    //             })
    //         })
    //         .collect::<Vec<_>>()
    // });
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

fn generations<L, N>(samples: &[RecExpr<L>], eqsat_results: &mut [EqsatResult<L, N>]) -> Vec<usize>
where
    L: Language + Display + FromOp,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    let generations = samples
        .iter()
        .map(|sample| {
            eqsat_results
                .iter_mut()
                .map(|r| r.egraph())
                .position(|egraph| egraph.lookup_expr(sample).is_some())
                .expect("Must be in at least one of the egraphs")
                + 1
        })
        .collect::<Vec<_>>();
    generations
}

// fn baselines<L, N>(
//     samples: &[RecExpr<L>],
//     eqsat_conf: &EqsatConf,
//     baseline_args: &BaselineArgs,
//     rules: &[Rewrite<L, N>],
//     rng: &mut ChaCha12Rng,
//     generations: &[usize],
//     goal_gen: usize,
// ) -> HashMap<usize, HashMap<usize, EqsatStats>>
// where
//     L: Language + Display + FromOp + 'static,
//     N: Analysis<L> + Clone + Default + Debug + 'static,
//     N::Data: Serialize + Clone,
// {
//     info!("Taking goals from generation {goal_gen}");
//     let random_goals = random_indices_eq(generations, goal_gen, baseline_args.random_goals(), rng);
//     let guide_gen = goal_gen / 2;
//     info!("Taking guides from generation {guide_gen}");
//     let random_guides =
//         random_indices_eq(generations, guide_gen, baseline_args.random_goals(), rng);

//     info!("Running goal-guide baselines...");
//     let baselines = random_guides
//         .iter()
//         .map(|guide_idx| {
//             let baseline = random_goals
//                 .iter()
//                 .map(|goal_idx| {
//                     let goal = samples[*goal_idx].to_owned();
//                     let guide = samples[*guide_idx].to_owned();
//                     info!("Running baseline for \"{goal}\" with guide \"{guide}\"...");
//                     let exprs = [&guide, &goal];
//                     let starting_exprs = StartMaterial::RecExprs(&exprs);
//                     let mut conf = eqsat_conf.to_owned();
//                     conf.root_check = true;
//                     conf.iter_limit = 100;
//                     let result = Eqsat::new(starting_exprs, rules).with_conf(conf).run();
//                     let baseline = result.into();
//                     info!("Baseline run!");
//                     (*goal_idx, baseline)
//                 })
//                 .collect();
//             (*guide_idx, baseline)
//         })
//         .collect();
//     info!("Goal-guide baselines run!");
//     baselines
// }

// fn random_indices_eq<T: PartialEq>(ts: &[T], t: T, n: usize, rng: &mut ChaCha12Rng) -> Vec<usize> {
//     ts.iter()
//         .enumerate()
//         .filter_map(
//             |(idx, generation)| {
//                 if *generation == t {
//                     Some(idx)
//                 } else {
//                     None
//                 }
//             },
//         )
//         .choose_multiple(rng, n)
// }

// fn explanation<L, N>(
//     cli: &Cli,
//     egraph: &mut EGraph<L, N>,
//     start_expr: &RecExpr<L>,
//     target_expr: &RecExpr<L>,
// ) -> Option<String>
// where
//     L: Language + Display + FromOp,
//     N: Analysis<L>,
// {
//     if cli.with_explanations() {
//         debug!("Constructing explanation of \"{start_expr} == {target_expr}\"...");
//         let mut expl = egraph.explain_equivalence(start_expr, target_expr);
//         let expl_chain = explanation::explanation_chain(&mut expl);
//         let flat_string = expl.get_flat_string();
//         debug!("Explanation constructed!");
//         Some(flat_string)
//     } else {
//         None
//     }
// }

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
