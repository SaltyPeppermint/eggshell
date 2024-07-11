#![warn(clippy::all, clippy::pedantic)]

use std::time::Duration;

use clap::Parser;

use eggshell::argparse::{CliArgs, EqsatArgs, Mode, Strategy};
use eggshell::baseline::{prove_expr, pulse_prove_expr, pulse_simplify_expr, simplify_expr};
use eggshell::io::reader;
use eggshell::io::writer;
use eggshell::trs::halide::Halide;
use eggshell::utils::AstSize2 as AstSize;

#[allow(dead_code)]
fn main() {
    fn main() {
        let args = CliArgs::parse();
        match args.mode {
            Mode::Prove {
                strategy,
                iteration_check,
            } => prove(&strategy, &args.args, iteration_check),
            Mode::Simplify { strategy } => simplify(&strategy, &args.args),
        }
    }
}

fn prove(strategy: &Strategy, args: &EqsatArgs, iter_check: bool) {
    match strategy {
        Strategy::Simple => {
            let exprs = reader::read_exprs(&args.expr_file).unwrap();
            let reports = exprs
                .iter()
                .map(|expr| prove_expr::<Halide, AstSize>(expr, args, iter_check))
                .collect::<Vec<_>>();
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
        Strategy::Pulse {
            time_limit,
            phase_limit,
        } => {
            let exprs = reader::read_exprs(&args.expr_file).unwrap();
            let reports = exprs
                .iter()
                .map(|expr| {
                    pulse_prove_expr::<Halide, _>(
                        expr,
                        Some(Duration::from_secs_f64(*time_limit)),
                        Some(*phase_limit),
                        args,
                        &AstSize,
                        iter_check,
                    )
                })
                .collect::<Vec<_>>();
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
    }
}

fn simplify(strategy: &Strategy, args: &EqsatArgs) {
    match strategy {
        Strategy::Simple => {
            let exprs = reader::read_exprs(&args.expr_file).unwrap();
            let reports = exprs
                .iter()
                .map(|expr| simplify_expr::<Halide, _>(expr, args, &AstSize))
                .collect::<Vec<_>>();
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
        Strategy::Pulse {
            time_limit,
            phase_limit,
        } => {
            let exprs = reader::read_exprs(&args.expr_file).unwrap();
            let reports = exprs
                .iter()
                .map(|expr| {
                    pulse_simplify_expr::<Halide, _>(
                        expr,
                        Some(Duration::from_secs_f64(*time_limit)),
                        Some(*phase_limit),
                        args,
                        &AstSize,
                    )
                })
                .collect::<Vec<_>>();
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
    }
}
