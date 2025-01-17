use core::panic;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::thread::available_parallelism;

use chrono::Local;
use clap::Parser;
use egg::{
    Analysis, AstSize, CostFunction, EGraph, FromOp, Language, RecExpr, Rewrite, StopReason,
};
use hashbrown::{HashMap, HashSet};
use log::{debug, info};
use num::BigUint;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

use eggshell::cli::{BaselineArgs, BaselineCmd, Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult, StartMaterial};
use eggshell::io::reader;
use eggshell::io::sampling::{DataEntry, EqsatStats, MetaData, SampleData};
use eggshell::io::structs::Entry;
use eggshell::sampling::strategy::{
    CostWeighted, CountWeightedGreedy, CountWeightedUniformly, Strategy,
};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Rise, TermRewriteSystem};
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

fn main() {
    env_logger::init();
    let start_time = Local::now();

    let cli = Cli::parse();

    let sample_conf = (&cli).into();
    let eqsat_conf = (&cli).into();

    let folder = format!(
        "data/generated_samples/{}/{}-{}-{}",
        cli.trs(),
        cli.file().file_stem().unwrap().to_str().unwrap(),
        start_time.format("%Y-%m-%d"),
        cli.uuid()
    );
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
                sample_conf,
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
                sample_conf,
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
    sample_conf: SampleConf,
    folder: String,
    timestamp: i64,
    rules: Vec<Rewrite<R::Language, R::Analysis>>,
    cli: &Cli,
) {
    let entry = exprs
        .into_iter()
        .nth(cli.expr_id())
        .expect("Must be in the file!");

    info!("Starting work on expr {}: {}...", cli.expr_id(), entry.expr);

    info!("Starting Eqsat...");

    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let mut eqsat_results = run_eqsats(&start_expr, &eqsat_conf, rules.as_slice());

    info!("Finished Eqsat!");
    info!("Starting sampling...");

    let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);

    let samples: Vec<_> = eqsat_results
        .iter()
        .enumerate()
        .flat_map(|(eqsat_generation, eqsat_result)| {
            info!("Running sampling of generation {}...", eqsat_generation + 1);
            let s = sample_eqsat_result(cli, &start_expr, eqsat_result, &sample_conf, &mut rng);
            info!("Finished sampling of generation {}!", eqsat_generation + 1);
            s
        })
        .collect();

    info!("Finished sampling!");
    info!(
        "Took {} unique samples while aiming for {}.",
        samples.len(),
        cli.eclass_samples() * eqsat_results.len()
    );

    let max_generation = eqsat_results.len();
    let generations = find_generations(&samples, &mut eqsat_results);

    let baselines = if let BaselineCmd::WithBaseline(args) = cli.baseline() {
        Some(mk_baselines(
            &samples,
            &eqsat_conf,
            args,
            rules.as_slice(),
            &mut rng,
            &generations,
            max_generation,
        ))
    } else {
        None
    };

    info!("Generating explanations...");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(available_parallelism().unwrap().into()) // Adjust if memory issues
        .build()
        .unwrap();
    let explenations: Vec<Option<_>> = pool.install(|| {
        samples
            .par_iter()
            .zip(generations.par_iter())
            .map_with(
                eqsat_results,
                |thread_local_eqsat_results, (sample, generation)| {
                    mk_explanation(
                        cli,
                        thread_local_eqsat_results[*generation - 1].egraph_mut(),
                        &start_expr,
                        sample,
                    )
                },
            )
            .collect()
    });

    info!("Finished generating explanations!");

    info!("Generating associated data...");
    let sample_data = samples
        .into_iter()
        .zip(explenations)
        .enumerate()
        .map(|(idx, (sample, explenation))| SampleData::new(sample, generations[idx], explenation))
        .collect();
    info!("Finished generating associated data!");

    info!("Finished work on expr {}!", cli.expr_id());

    let data = DataEntry::new(
        start_expr,
        sample_data,
        baselines,
        MetaData::new(
            cli.uuid().to_owned(),
            folder.to_owned(),
            cli.to_owned(),
            timestamp,
            sample_conf,
            eqsat_conf,
            rules.into_iter().map(|r| r.name.to_string()).collect(),
        ),
    );

    let mut f = BufWriter::new(File::create(format!("{folder}/{}.json", cli.expr_id())).unwrap());
    serde_json::to_writer(&mut f, &data).unwrap();
    f.flush().unwrap();

    info!("Results for expr {} written to disk!", cli.expr_id());

    // });
}

fn run_eqsats<L, N>(
    start_expr: &RecExpr<L>,
    eqsat_conf: &EqsatConf,
    rules: &[Rewrite<L, N>],
) -> Vec<EqsatResult<L, N>>
where
    L: Language + Display + Serialize + 'static,
    N: Analysis<L> + Clone + Serialize + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    let mut eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.to_owned()]), rules)
        .with_conf(eqsat_conf.to_owned());
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
fn sample_eqsat_result<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    eqsat: &EqsatResult<L, N>,
    sample_conf: &SampleConf,
    rng: &mut ChaCha12Rng,
) -> HashSet<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let root_id = eqsat.roots()[0];

    match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeightedUniformly::<BigUint, _, _>::new_with_limit(eqsat.egraph(), limit)
                .sample_eclass(rng, sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CountWeighted => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeightedGreedy::<BigUint, _, _>::new_with_limit(eqsat.egraph(), start_expr, limit)
                .sample_eclass(rng, sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CostWeighted => {
            CostWeighted::new(eqsat.egraph(), AstSize, sample_conf.loop_limit)
                .sample_eclass(rng, sample_conf, root_id)
                .unwrap()
        }
    }
}

fn find_generations<L, N>(
    samples: &[RecExpr<L>],
    eqsat_results: &mut [EqsatResult<L, N>],
) -> Vec<usize>
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

fn mk_baselines<L, N>(
    samples: &[RecExpr<L>],
    eqsat_conf: &EqsatConf,
    baseline_args: &BaselineArgs,
    rules: &[Rewrite<L, N>],
    rng: &mut ChaCha12Rng,
    generations: &[usize],
    goal_gen: usize,
) -> HashMap<usize, HashMap<usize, EqsatStats>>
where
    L: Language + Display + FromOp + 'static,
    N: Analysis<L> + Clone + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    info!("Taking goals from generation {goal_gen}");
    let random_goals = random_indices_eq(generations, goal_gen, baseline_args.random_goals(), rng);
    let guide_gen = goal_gen / 2;
    info!("Taking guides from generation {guide_gen}");
    let random_guides =
        random_indices_eq(generations, guide_gen, baseline_args.random_goals(), rng);

    info!("Running goal-guide baselines...");
    let baselines = random_guides
        .iter()
        .map(|guide_idx| {
            let baseline = random_goals
                .iter()
                .map(|goal_idx| {
                    let goal = samples[*goal_idx].to_owned();
                    let guide = samples[*guide_idx].to_owned();
                    info!("Running baseline for \"{goal}\" with guide \"{guide}\"...");
                    let starting_exprs = StartMaterial::RecExprs(vec![guide, goal]);
                    let mut conf = eqsat_conf.to_owned();
                    conf.root_check = true;
                    conf.iter_limit = 100;
                    let result = Eqsat::new(starting_exprs, rules).with_conf(conf).run();
                    let baseline = result.into();
                    info!("Baseline run!");
                    (*goal_idx, baseline)
                })
                .collect();
            (*guide_idx, baseline)
        })
        .collect();
    info!("Goal-guide baselines run!");
    baselines
}

fn random_indices_eq<T: PartialEq>(ts: &[T], t: T, n: usize, rng: &mut ChaCha12Rng) -> Vec<usize> {
    ts.iter()
        .enumerate()
        .filter_map(
            |(idx, generation)| {
                if *generation == t {
                    Some(idx)
                } else {
                    None
                }
            },
        )
        .choose_multiple(rng, n)
}

fn mk_explanation<L, N>(
    cli: &Cli,
    egraph: &mut EGraph<L, N>,
    start_expr: &RecExpr<L>,
    target_expr: &RecExpr<L>,
) -> Option<String>
where
    L: Language + Display + FromOp,
    N: Analysis<L>,
{
    if cli.with_explanations() {
        debug!("Constructing explanation of \"{start_expr} == {target_expr}\"...");
        let expl = egraph
            .explain_equivalence(start_expr, target_expr)
            .get_flat_string();
        debug!("Explanation constructed!");
        Some(expl)
    } else {
        None
    }
}
