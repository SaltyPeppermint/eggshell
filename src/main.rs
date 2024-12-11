use core::panic;
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

use chrono::Local;
use clap::Parser;
use egg::{
    Analysis, AstSize, CostFunction, EGraph, FromOp, Language, RecExpr, Rewrite, StopReason,
};
use hashbrown::HashSet;
use log::info;
use num::BigUint;
use rand::SeedableRng;

use eggshell::cli::{Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{Eqsat, EqsatConf, EqsatResult, StartMaterial};
use eggshell::io::reader;
use eggshell::io::sampling::{BaselineData, DataEntry, MetaData, SampleData};
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
            run_eqsat::<Halide>(
                exprs,
                eqsat_conf,
                sample_conf,
                folder,
                Halide::full_rules().as_slice(),
                &cli,
            );
        }
        TrsName::Rise => {
            run_eqsat::<Rise>(
                exprs,
                eqsat_conf,
                sample_conf,
                folder,
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

fn run_eqsat<R: TermRewriteSystem>(
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
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
    let mut eqsat =
        Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()])).with_conf(eqsat_conf.clone());

    let mut eqsat_results = Vec::new();
    let mut iter_count = 0;

    loop {
        let result = eqsat.run(rules);
        iter_count += 1;
        info!("Iteration {iter_count} finished.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                eqsat_results.push(result.clone());
                eqsat = Eqsat::new(result.into()).with_conf(eqsat_conf.clone());
            }
            _ => {
                info!("Limits reached after {} full iterations!", iter_count - 1);
                //
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

    info!("Finished Eqsat {}!", cli.expr_id());
    info!("Starting sampling...");

    let samples: Vec<_> = eqsat_results
        .iter()
        .enumerate()
        .flat_map(|(eqsat_round, eqsat_result)| {
            info!("Running sampling of round {eqsat_round}...");
            let s = sample(cli, &start_expr, eqsat_result, &sample_conf);
            info!("Finished sampling of round {eqsat_round}!");
            s
        })
        .collect();

    info!("Finished sampling {}!", cli.expr_id());
    info!(
        "Took {} unique samples while aiming for {}.",
        samples.len(),
        cli.eclass_samples()
    );

    info!("Generating associated data for {}...", cli.expr_id());
    let sample_data = mk_sample_data(&start_expr, samples, &mut eqsat_results, cli, rules);
    info!("Finished generating sample data for {}!", cli.expr_id());

    info!("Finished work on expr {}!", cli.expr_id());

    let data = DataEntry::new(
        start_expr,
        sample_data,
        MetaData::new(
            cli.uuid().to_owned(),
            folder.to_owned(),
            cli.to_owned(),
            Local::now().timestamp(),
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

/// Inner sample logic.
/// Samples guranteed to be unique.
fn sample<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    eqsat: &EqsatResult<L, N>,
    sample_conf: &SampleConf,
) -> HashSet<RecExpr<L>>
where
    L: Language + Display + Clone + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Clone + Debug + Sync,
    N::Data: Serialize + Clone + Sync,
{
    let root_id = eqsat.roots()[0];
    let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);

    match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeightedUniformly::<BigUint, _, _>::new_with_limit(eqsat.egraph(), limit)
                .sample_eclass(&mut rng, sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CountWeighted => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeighted::<BigUint, _, _>::new_with_limit(eqsat.egraph(), start_expr, limit)
                .sample_eclass(&mut rng, sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CostWeighted => {
            CostWeighted::new(eqsat.egraph(), AstSize, sample_conf.loop_limit)
                .sample_eclass(&mut rng, sample_conf, root_id)
                .unwrap()
        }
    }
}

fn mk_sample_data<L, N>(
    start_expr: &RecExpr<L>,
    samples: Vec<RecExpr<L>>,
    eqsat_result: &mut [EqsatResult<L, N>],
    cli: &Cli,
    rules: &[Rewrite<L, N>],
) -> Vec<SampleData<L>>
where
    L: Language + Display + Clone + FromOp,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    samples
        .into_iter()
        .map(|sample| {
            let baseline = mk_baseline(cli, start_expr, &sample, rules);
            let explanation = mk_explanation(
                &sample,
                cli,
                eqsat_result.last_mut().unwrap().egraph_mut(),
                start_expr,
            );
            let generation =
                find_generation(&sample, &mut eqsat_result.iter_mut().map(|r| r.egraph()));
            SampleData::new(sample, generation, baseline, explanation)
        })
        .collect()
}

fn mk_baseline<L, N>(
    cli: &Cli,
    start_expr: &RecExpr<L>,
    sample: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
) -> Option<BaselineData>
where
    L: Language + Display + Clone,
    N: Analysis<L> + Clone + Default + Debug,
    N::Data: Serialize + Clone,
{
    if cli.with_baselines() {
        info!("Running baseline for \"{sample}\"...");
        let starting_exprs = StartMaterial::RecExprs(vec![start_expr.clone(), sample.clone()]);
        let result = Eqsat::new(starting_exprs).run(rules);
        let b = BaselineData::new(
            result.report().stop_reason.clone(),
            result.report().total_time,
            result.report().egraph_nodes,
            result.report().iterations,
        );
        info!("Baseline run!");
        Some(b)
    } else {
        None
    }
}

fn mk_explanation<L, N>(
    sample: &RecExpr<L>,
    cli: &Cli,
    egraph: &mut EGraph<L, N>,
    start_expr: &RecExpr<L>,
) -> Option<String>
where
    L: Language + Display + FromOp,
    N: Analysis<L>,
{
    if cli.with_explanations() {
        info!("Constructing explanation of \"{start_expr} == {sample}\"...");
        let expl = egraph
            .explain_equivalence(start_expr, sample)
            .get_flat_string();
        info!("Explanation constructed!");
        Some(expl)
    } else {
        None
    }
}

fn find_generation<'a, L: Language + 'a, N: Analysis<L> + 'a>(
    sample: &RecExpr<L>,
    egraphs: &mut impl ExactSizeIterator<Item = &'a EGraph<L, N>>,
) -> usize {
    match egraphs.position(|egraph| egraph.lookup_expr(sample).is_some()) {
        Some(generation) => generation + 1,
        None => egraphs.len(),
    }
}
