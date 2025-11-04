use core::panic;
use std::fmt::{Debug, Display};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local, TimeDelta};
use clap::Parser;
use egg::{Analysis, FromOp, Id, Language, RecExpr, Rewrite, Runner, SimpleScheduler, StopReason};
use eggshell::rewrite_system::{dummy_rise, halide};
use eggshell::sampling::SampleError;
use log::{debug, info};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use eggshell::cli::{Cli, RewriteSystemName};
use eggshell::eqsat::{self, EqsatConf, StartMaterial};
use eggshell::io::{reader, structs::Entry};
use eggshell::sampling::sampler::{Greedy, Sampler};

fn main() {
    env_logger::init();
    let start_time = Local::now();
    let cli = Cli::parse();
    let file_stem_str = cli.file().file_stem().unwrap().to_str().unwrap();
    let folder = PathBuf::from("data/generated_samples")
        .join(cli.rewrite_system().to_string())
        .join(format!("{file_stem_str}-{}", start_time.to_rfc3339()));
    fs::create_dir_all(&folder).unwrap();

    let exprs = match cli.file().extension().unwrap().to_str().unwrap() {
        "csv" => reader::read_exprs_csv(cli.file()),
        "json" => reader::read_exprs_json(cli.file()),
        extension => panic!("Unknown file extension {}", extension),
    };

    let entry = &exprs[cli.expr_id()];
    println!("Starting work on expr {}: {}...", cli.expr_id(), entry.expr);

    let term_folder = folder.join(cli.expr_id().to_string());
    fs::create_dir_all(&term_folder).unwrap();
    println!("Will write to folder: {}", term_folder.to_string_lossy());

    match cli.rewrite_system() {
        RewriteSystemName::Halide => {
            run(
                entry,
                &halide::rules(halide::HalideRuleset::Full),
                &term_folder,
                start_time,
                &cli,
            );
        }
        RewriteSystemName::Rise => {
            run(
                entry,
                &dummy_rise::full_rules(),
                &term_folder,
                start_time,
                &cli,
            );
        }
    }

    print_delta(Local::now() - start_time);
    println!("Work on expression {} done!", cli.expr_id());
}

fn run<L, N>(
    entry: &Entry,
    rules: &[Rewrite<L, N>],
    folder: &Path,
    start_time: DateTime<Local>,
    cli: &Cli,
) where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    let start_expr = entry.expr.parse::<RecExpr<L>>().unwrap();
    let rule_names = rules.iter().map(|r| r.name.to_string()).collect::<Vec<_>>();
    let size_limit = start_expr.len() * 2;

    let metadata = MetaData::new(cli, &start_time, &rule_names);
    let rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());

    let (chain_lengths, stop_reasons) = (0..cli.n_chains())
        .into_par_iter()
        .map_with(rng, |thread_rng, chain_id| {
            thread_rng.set_word_pos(0); // For reproducibility
            thread_rng.set_stream(chain_id);
            let (chain, outcome) = chain_rec(
                cli,
                rules,
                size_limit,
                thread_rng,
                chain_id,
                vec![start_expr.clone()],
            );
            let data = DataEntry::new(cli.iter_distance(), chain.as_slice(), &metadata);
            save(folder, chain_id, &data);
            (chain.len(), outcome)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let total_samples = chain_lengths.iter().sum::<usize>();
    #[expect(clippy::cast_precision_loss)]
    let average_samples = total_samples as f64 / chain_lengths.len() as f64;
    let ended_with_error = stop_reasons.iter().filter(|x| x.is_some()).count();

    println!("Total Samples: {total_samples}");
    println!("Average Samples per Chain: {average_samples}");
    println!("Ended early: {ended_with_error}");
}

fn chain_rec<L, N>(
    cli: &Cli,
    rules: &[Rewrite<L, N>],
    size_limit: usize,
    rng: &mut ChaCha12Rng,
    chain_id: u64,
    mut chain: Vec<RecExpr<L>>,
) -> (Vec<RecExpr<L>>, Option<SampleError>)
where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    if chain.len() > cli.chain_length() {
        return (chain, None);
    }

    match sample(rng, rules, chain.last().unwrap(), cli, size_limit) {
        Ok(sample) => {
            chain.push(sample);
            chain_rec(cli, rules, size_limit, rng, chain_id, chain)
        }
        Err(e) => {
            debug!(
                "Chain {chain_id} ended after {} terms with error: {e}",
                chain.len()
            );
            (chain, Some(e))
        }
    }
}

fn sample<L, N>(
    rng: &mut ChaCha12Rng,
    rules: &[Rewrite<L, N>],
    start_expr: &RecExpr<L>,
    cli: &Cli,
    size_limit: usize,
) -> Result<RecExpr<L>, SampleError>
where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    info!("Starting Eqsat...");
    let (eqsat, penultimate_eqsat) = run_eqsat(start_expr, rules, cli.iter_distance())?;
    info!("Finished Eqsat!");
    let sampler = Greedy::new(&eqsat.0.egraph);

    info!("Starting sampling...");
    for i in 0..cli.max_retries() {
        let sample = sampler.sample_expr(rng, &eqsat.0.egraph[eqsat.1[0]], size_limit)?;
        if penultimate_eqsat.0.egraph.lookup_expr(&sample).is_none() {
            info!("Sample found after {i} tries...");
            return Ok(sample);
        }
    }
    Err(SampleError::RetryLimit(cli.max_retries()))
}

#[expect(clippy::type_complexity)]
fn run_eqsat<L, N>(
    start_expr: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
    iter_distance: usize,
) -> Result<((Runner<L, N>, Vec<Id>), (Runner<L, N>, Vec<Id>)), SampleError>
where
    L: Language + Display + Serialize + 'static,
    N: Analysis<L> + Clone + Serialize + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    let penultimate_result = eqsat::eqsat(
        &EqsatConf::builder().iter_limit(iter_distance - 1).build(),
        start_expr.into(),
        rules,
        None,
        SimpleScheduler,
    );

    if let StopReason::IterationLimit(i) = penultimate_result.0.report().stop_reason {
        if i < iter_distance - 1 {
            return Err(SampleError::IterDistance(i));
        }
    } else {
        return Err(SampleError::OtherStop(
            penultimate_result.0.report().stop_reason.clone(),
        ));
    }
    let result = eqsat::eqsat(
        &EqsatConf::builder().iter_limit(1).build(),
        StartMaterial::from_egraph_and_roots(
            penultimate_result.0.egraph.clone(),
            penultimate_result.1.clone(),
        ),
        rules,
        None,
        SimpleScheduler,
    );
    if let StopReason::IterationLimit(_) = &result.0.report().stop_reason {
        return Ok((result, penultimate_result));
    }
    Err(SampleError::OtherStop(
        result.0.report().stop_reason.clone(),
    ))
}

fn save<L: Language + FromOp + Display + Serialize>(
    term_folder: &Path,
    chain_id: u64,
    data: &DataEntry<L>,
) {
    let batch_file = term_folder
        .join(chain_id.to_string())
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
    iterations: usize,
    chain: &'a [RecExpr<L>],
    metadata: &'a MetaData<'a>,
}

impl<'a, L: Language + FromOp + Display> DataEntry<'a, L> {
    #[must_use]
    pub fn new(iterations: usize, chain: &'a [RecExpr<L>], metadata: &'a MetaData<'a>) -> Self {
        Self {
            iterations,
            chain,
            metadata,
        }
    }
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData<'a> {
    cli: &'a Cli,
    start_time: String,
    rule_names: &'a [String],
}

impl<'a> MetaData<'a> {
    #[must_use]
    pub fn new(cli: &'a Cli, start_time: &DateTime<Local>, rule_names: &'a [String]) -> Self {
        Self {
            cli,
            start_time: start_time.to_rfc3339(),
            rule_names,
        }
    }
}
