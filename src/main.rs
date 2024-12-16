use core::panic;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

use chrono::Local;
use clap::Parser;
use egg::{Analysis, AstSize, CostFunction, FromOp, Language, RecExpr, Rewrite, StopReason};
use hashbrown::{HashMap, HashSet};
use log::info;
use num::BigUint;
use rand::seq::IteratorRandom;
use rand::SeedableRng;

use eggshell::cli::{BaselineArgs, Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult, StartMaterial};
use eggshell::io::reader;
use eggshell::io::sampling::{DataEntry, EqsatStats, MetaData, SampleData};
use eggshell::io::structs::Entry;
use eggshell::sampling::strategy::{CostWeighted, CountWeighted, CountWeightedUniformly, Strategy};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Rise, TermRewriteSystem};
use rand_chacha::ChaCha12Rng;
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
                Halide::full_rules().as_slice(),
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
                Rise::full_rules().as_slice(),
                &cli,
            );
        }
    }

    let runtime = Local::now() - start_time;
    info!(
        "Runtime: {}:{}:{}",
        runtime.num_hours(),
        runtime.num_minutes(),
        runtime.num_seconds()
    );

    info!("EXPR {} DONE!", cli.expr_id());
}

fn run<R: TermRewriteSystem>(
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
    timestamp: i64,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: &Cli,
) {
    let entry = exprs
        .into_iter()
        .nth(cli.expr_id())
        .expect("Must be in the file!");

    info!(
        "Starting work on expression {}: {}...",
        cli.expr_id(),
        entry.expr
    );

    info!("Starting eqsat on expression {}...", cli.expr_id());

    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let mut eqsat_results = run_eqsats(&start_expr, &eqsat_conf, rules);

    info!("Finished Eqsat {}!", cli.expr_id());
    info!("Starting sampling...");

    let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);

    let samples: Vec<_> = eqsat_results
        .iter()
        .enumerate()
        .flat_map(|(eqsat_generation, eqsat_result)| {
            info!("Running sampling of generation {eqsat_generation}...");
            let s = sample_eqsat_result(cli, &start_expr, eqsat_result, &sample_conf, &mut rng);
            info!("Finished sampling of generation {eqsat_generation}!");
            s
        })
        .collect();

    info!("Finished sampling {}!", cli.expr_id());
    info!(
        "Took {} unique samples while aiming for {}.",
        samples.len(),
        cli.eclass_samples()
    );

    let max_generation = eqsat_results.len();
    let generations = find_generations(&samples, &mut eqsat_results);
    drop(eqsat_results);

    let baselines = cli.baseline_args().map(|baselin_args| {
        mk_baselines(
            &samples,
            &eqsat_conf,
            baselin_args,
            rules,
            &mut rng,
            &generations,
            max_generation,
        )
    });

    info!("Generating associated data for {}...", cli.expr_id());
    let sample_data = samples
        .into_iter()
        .enumerate()
        .map(|(idx, sample)| SampleData::new(sample, generations[idx]))
        .collect();
    info!("Finished generating sample data for {}!", cli.expr_id());

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
    L: Language + Display + Serialize,
    N: Analysis<L> + Clone + Serialize + Default + Debug,
    N::Data: Serialize + Clone,
{
    let mut eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.to_owned()]))
        .with_conf(eqsat_conf.to_owned());

    let mut eqsat_results = Vec::new();
    let mut iter_count = 0;

    loop {
        let result = eqsat.run(rules);
        iter_count += 1;
        info!("Iteration {iter_count} stopped.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                eqsat_results.push(result.clone());
                eqsat = Eqsat::new(result.into()).with_conf(eqsat_conf.to_owned());
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
            CountWeighted::<BigUint, _, _>::new_with_limit(eqsat.egraph(), start_expr, limit)
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
    L: Language + Display + Clone + FromOp,
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
    L: Language + Display + Clone + FromOp,
    N: Analysis<L> + Clone + Default + Debug,
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
                    let result = Eqsat::new(starting_exprs).with_conf(conf).run(rules);
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

// fn mk_explanation<L, N>(
//     sample: &RecExpr<L>,
//     cli: &Cli,
//     egraph: &mut EGraph<L, N>,
//     start_expr: &RecExpr<L>,
// ) -> Option<String>
// where
//     L: Language + Display + FromOp,
//     N: Analysis<L>,
// {
//     if cli.with_explanations() {
//         info!("Constructing explanation of \"{start_expr} == {sample}\"...");
//         let expl = egraph
//             .explain_equivalence(start_expr, sample)
//             .get_flat_string();
//         info!("Explanation constructed!");
//         Some(expl)
//     } else {
//         None
//     }
// }
