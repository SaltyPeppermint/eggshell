use core::panic;
use std::fmt::{Debug, Display};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local, TimeDelta};
use clap::Parser;
use egg::{Analysis, FromOp, Language, RecExpr, Rewrite, SimpleScheduler, StopReason};
use eggshell::sampling::SampleError;
use log::{info, warn};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use serde::Serialize;

use eggshell::cli::{Cli, RewriteSystemName};
use eggshell::eqsat::{self, EqsatConf, EqsatResult};
use eggshell::io::{reader, structs::Entry};
use eggshell::rewrite_system::{Halide, RewriteSystem, Rise};
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
            run::<Halide>(entry, &term_folder, start_time, &cli);
        }
        RewriteSystemName::Rise => {
            run::<Rise>(entry, &term_folder, start_time, &cli);
        }
    }

    print_delta(Local::now() - start_time);
    println!("Work on expression {} done!", cli.expr_id());
}

fn run<R: RewriteSystem>(entry: &Entry, folder: &Path, start_time: DateTime<Local>, cli: &Cli) {
    let start_expr = entry.expr.parse::<RecExpr<R::Language>>().unwrap();
    let rules = R::full_rules();
    let rule_names = rules.iter().map(|r| r.name.to_string()).collect::<Vec<_>>();
    let size_limit = start_expr.len() * 2;

    let metadata = MetaData::new(cli, &start_time, &rule_names, &start_expr);
    let rng = ChaCha12Rng::seed_from_u64(cli.rng_seed());

    let (chain_lengths, stop_reasons) = (0..cli.n_chains())
        .into_par_iter()
        .map_with(rng, |thread_rng, chain_id| {
            thread_rng.set_word_pos(0); // For reproducibility
            thread_rng.set_stream(chain_id);
            let mut chain = vec![start_expr.clone()];
            continue_chain(
                folder, cli, &rules, size_limit, &metadata, thread_rng, chain_id, &mut chain, 0,
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let total_samples = chain_lengths.iter().sum::<usize>();
    let average_samples = total_samples / chain_lengths.len();
    let ended_with_error = stop_reasons.iter().filter(|x| x.is_some()).count();

    println!("Total Samples: {total_samples}");
    println!("Average Samples per Chain: {average_samples}");
    println!("Ended early: {ended_with_error}");
}

#[expect(clippy::too_many_arguments)]
fn continue_chain<L, N>(
    folder: &Path,
    cli: &Cli,
    rules: &[Rewrite<L, N>],
    size_limit: usize,
    metadata: &MetaData<'_, L>,
    thread_rng: &mut ChaCha12Rng,
    chain_id: u64,
    chain: &mut Vec<RecExpr<L>>,
    i: usize,
) -> (usize, Option<SampleError>)
where
    L: Language + Display + FromOp + Clone + Send + Sync + Serialize + 'static,
    L::Discriminant: Send + Sync,
    N: Analysis<L> + Clone + Debug + Default + Send + Sync + Serialize + 'static,
    N::Data: Serialize + Clone + Send + Sync,
{
    if i > cli.chain_length() {
        return (i, None);
    }

    let sample = match sample(thread_rng, rules, chain.last().unwrap(), cli, size_limit) {
        Ok(sample) => sample,
        Err(e) => {
            warn!("Chain {chain_id} ended after {i} terms with error: {e}");
            let data = DataEntry::new(cli.iter_distance(), chain.as_slice(), metadata);
            save_batch(folder, chain_id, i / cli.batch_size(), &data);
            return (i, Some(e));
        }
    };
    chain.push(sample);
    if chain.len() >= cli.batch_size() {
        let data = DataEntry::new(cli.iter_distance(), chain.as_slice(), metadata);
        save_batch(folder, chain_id, i / cli.batch_size(), &data);
        *chain = vec![chain.pop().unwrap()];
    }

    continue_chain(
        folder,
        cli,
        rules,
        size_limit,
        metadata,
        thread_rng,
        chain_id,
        chain,
        i + 1,
    )
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
    let sampler = Greedy::new(eqsat.egraph());

    info!("Starting sampling...");
    for i in 0..cli.max_retries() {
        let sample = sampler.sample_expr(rng, &eqsat.egraph()[eqsat.roots()[0]], size_limit)?;
        if penultimate_eqsat.egraph().lookup_expr(&sample).is_none() {
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
) -> Result<(EqsatResult<L, N>, EqsatResult<L, N>), SampleError>
where
    L: Language + Display + Serialize + 'static,
    N: Analysis<L> + Clone + Serialize + Default + Debug + 'static,
    N::Data: Serialize + Clone,
{
    let penultimate_result = eqsat::eqsat(
        &EqsatConf::builder()
            .iter_limit(iter_distance - 1) // Important to capture the egraph after every iteration!
            .build(),
        start_expr.into(),
        rules,
        None,
        SimpleScheduler,
    );

    let penultimate_stop_reason = &penultimate_result.report().stop_reason;
    match penultimate_stop_reason {
        StopReason::IterationLimit(i) => {
            if *i < iter_distance - 1 {
                return Err(SampleError::IterDistance(*i));
            }
        }
        _ => return Err(SampleError::OtherStop(penultimate_stop_reason.clone())),
    }

    let result = eqsat::eqsat(
        &EqsatConf::builder()
            .iter_limit(1) // Important to capture the egraph after every iteration!
            .build(),
        penultimate_result.clone().into(),
        rules,
        None,
        SimpleScheduler,
    );

    match &result.report().stop_reason {
        StopReason::IterationLimit(_) => Ok((result, penultimate_result)),
        _ => Err(SampleError::OtherStop(result.report().stop_reason.clone())),
    }
}

fn save_batch<L: Language + FromOp + Display + Serialize>(
    term_folder: &Path,
    chain_id: u64,
    batch_id: usize,
    data: &DataEntry<L>,
) {
    let batch_file = term_folder
        .join(format!("{chain_id}-{batch_id}"))
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
    metadata: &'a MetaData<'a, L>,
}

impl<'a, L: Language + FromOp + Display> DataEntry<'a, L> {
    #[must_use]
    pub fn new(iterations: usize, chain: &'a [RecExpr<L>], metadata: &'a MetaData<'a, L>) -> Self {
        Self {
            iterations,
            chain,
            metadata,
        }
    }
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq)]
pub struct MetaData<'a, L: Language + Display> {
    cli: &'a Cli,
    start_time: String,
    rule_names: &'a [String],
    start_expr: &'a RecExpr<L>,
}

impl<'a, L: Language + Display> MetaData<'a, L> {
    #[must_use]
    pub fn new(
        cli: &'a Cli,
        start_time: &DateTime<Local>,
        rules: &'a [String],
        start_expr: &'a RecExpr<L>,
    ) -> Self {
        Self {
            cli,
            start_time: start_time.to_rfc3339(),
            rule_names: rules,
            start_expr,
        }
    }
}
