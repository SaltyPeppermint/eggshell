use core::panic;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};

use chrono::Local;
use clap::Parser;
use egg::{Analysis, AstSize, CostFunction, EGraph, Language, RecExpr, Rewrite, StopReason};
use hashbrown::HashSet;
use log::info;
use num::BigUint;
use rand::rngs::StdRng;
use rand::SeedableRng;

use eggshell::cli::{Cli, SampleStrategy, TrsName};
use eggshell::eqsat::{EqsatConf, StartMaterial};
use eggshell::io::reader;
use eggshell::io::sampling::{BaselineData, DataEntry, MetaData, SampleData};
use eggshell::io::structs::Entry;
use eggshell::sampling::strategy::{CostWeighted, CountWeighted, CountWeightedUniformly, Strategy};
use eggshell::sampling::SampleConf;
use eggshell::trs::{Halide, Rise, TermRewriteSystem, TrsEqsat, TrsEqsatResult};

fn main() {
    env_logger::init();

    let cli = Cli::parse();

    let sample_conf = (&cli).into();
    let eqsat_conf = (&cli).into();
    let now = Local::now();

    let folder = format!(
        "data/generated_samples/{}/{}-{}-{}",
        cli.trs(),
        cli.file().file_stem().unwrap().to_str().unwrap(),
        now.format("%Y-%m-%d"),
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
                cli,
            );
        }
        TrsName::Rise => {
            run_eqsat::<Rise>(
                exprs,
                eqsat_conf,
                sample_conf,
                folder,
                Rise::full_rules().as_slice(),
                cli,
            );
        }
    }
}

fn run_eqsat<R: TermRewriteSystem>(
    exprs: Vec<Entry>,
    eqsat_conf: EqsatConf,
    sample_conf: SampleConf,
    folder: String,
    rules: &[Rewrite<R::Language, R::Analysis>],
    cli: Cli,
) {
    let rng = StdRng::seed_from_u64(sample_conf.rng_seed);

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
    let mut eqsat = TrsEqsat::<R>::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
        .with_conf(eqsat_conf.clone());

    let mut intermediate_egraphs = Vec::new();
    let mut iter_count = 0;

    let last_result = loop {
        let result = eqsat.run(rules);
        iter_count += 1;
        info!("Iteration {iter_count} finished.");

        assert!(result.egraph().clean);
        match result.report().stop_reason {
            StopReason::IterationLimit(_) => {
                intermediate_egraphs.push(result.egraph().clone());
                eqsat = TrsEqsat::<R>::new(result.into()).with_conf(eqsat_conf.clone());
            }
            _ => {
                info!("Limits reached after {iter_count} iterations!");
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

    info!("Finished Eqsat {}!", cli.expr_id());
    info!("Starting sampling...");

    let samples: Vec<_> = sample::<R>(&cli, &start_expr, &last_result, rng, &sample_conf)
        .into_iter()
        .collect();

    info!("Finished sampling {}!", cli.expr_id());
    info!(
        "Took {} unique samples while aiming for {}.",
        samples.len(),
        cli.eclass_samples()
    );

    info!("Generating associated data for {}...", cli.expr_id());
    let sample_data = mk_sample_data::<R>(
        &start_expr,
        samples,
        intermediate_egraphs.as_slice(),
        &cli,
        last_result,
        rules,
    );
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

    info!("All written to disk, done!");

    info!("EXPR {} IS DONE!", cli.expr_id())
    // });
}

/// Inner sample logic.
/// Samples guranteed to be unique.
fn sample<R: TermRewriteSystem>(
    cli: &Cli,
    start_expr: &RecExpr<R::Language>,
    eqsat: &TrsEqsatResult<R>,
    mut rng: StdRng,
    sample_conf: &SampleConf,
) -> HashSet<RecExpr<<R as TermRewriteSystem>::Language>> {
    let root_id = eqsat.roots()[0];

    match &cli.strategy() {
        SampleStrategy::CountWeightedUniformly => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeightedUniformly::<BigUint, _, _>::new_with_limit(eqsat.egraph(), &mut rng, limit)
                .sample_eclass(sample_conf, root_id)
                .unwrap()
        }
        SampleStrategy::CountWeighted => {
            let limit = (AstSize.cost_rec(start_expr) as f64 * 2.0) as usize;
            CountWeighted::<BigUint, _, _>::new_with_limit(
                eqsat.egraph(),
                &mut rng,
                start_expr,
                limit,
            )
            .sample_eclass(sample_conf, root_id)
            .unwrap()
        }
        SampleStrategy::CostWeighted => {
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit)
                .sample_eclass(sample_conf, root_id)
                .unwrap()
        }
    }
}

fn mk_sample_data<R: TermRewriteSystem>(
    start_expr: &RecExpr<R::Language>,
    samples: Vec<RecExpr<R::Language>>,
    intermediate_egraphs: &[EGraph<R::Language, R::Analysis>],
    cli: &Cli,
    mut eqsat: TrsEqsatResult<R>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Vec<SampleData<R::Language>> {
    samples
        .into_iter()
        .map(|sample| {
            let baseline = mk_baseline::<R>(cli, start_expr, &sample, rules);
            let explanation = mk_explanation::<R>(&sample, cli, &mut eqsat, start_expr);
            let generation = find_generation(&sample, intermediate_egraphs);
            SampleData::new(sample, generation, baseline, explanation)
        })
        .collect()
}

fn mk_baseline<R: TermRewriteSystem>(
    cli: &Cli,
    start_expr: &RecExpr<R::Language>,
    sample: &RecExpr<R::Language>,
    rules: &[Rewrite<R::Language, R::Analysis>],
) -> Option<BaselineData> {
    if cli.with_baselines() {
        info!("Running baseline for \"{sample}\"...");
        let starting_exprs = StartMaterial::RecExprs(vec![start_expr.clone(), sample.clone()]);
        let result = TrsEqsat::<R>::new(starting_exprs).run(rules);
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

fn mk_explanation<R: TermRewriteSystem>(
    sample: &RecExpr<R::Language>,
    cli: &Cli,
    eqsat: &mut TrsEqsatResult<R>,
    start_expr: &RecExpr<R::Language>,
) -> Option<String> {
    if cli.with_explanations() {
        info!("Constructing explanation of \"{start_expr} == {sample}\"...");
        let expl = eqsat
            .egraph_mut()
            .explain_equivalence(start_expr, sample)
            .get_flat_string();
        info!("Explanation constructed!");
        Some(expl)
    } else {
        None
    }
}

fn find_generation<L: Language, N: Analysis<L>>(
    sample: &RecExpr<L>,
    intermediate_egraphs: &[EGraph<L, N>],
) -> usize {
    match intermediate_egraphs
        .iter()
        .position(|egraph| egraph.lookup_expr(sample).is_some())
    {
        Some(generation) => generation,
        None => intermediate_egraphs.len(),
    }
}
