#![warn(clippy::all, clippy::pedantic)]

use std::time::Duration;

use clap::Parser;

use eggshell::argparse::{CliArgs, EqsatArgs, Mode, Strategy};
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
            let expression_vect = reader::read_expressions(&args.expressions_file).unwrap();
            let reports = eggshell::baseline::prove_expressions::<Halide, AstSize>(
                &expression_vect,
                args,
                iter_check,
            );
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
        Strategy::Pulse {
            time_limit,
            phase_limit,
        } => {
            let expression_vect = reader::read_expressions(&args.expressions_file).unwrap();
            let reports = eggshell::baseline::pulse_prove_expressions::<Halide, _>(
                &AstSize,
                &expression_vect,
                args,
                Some(Duration::from_secs_f64(*time_limit)),
                Some(*phase_limit),
                iter_check,
            );
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
    }
}

fn simplify(strategy: &Strategy, args: &EqsatArgs) {
    match strategy {
        Strategy::Simple => {
            let expression_vect = reader::read_expressions(&args.expressions_file).unwrap();
            let reports = eggshell::baseline::simplify_expressions::<Halide, _>(
                &AstSize,
                &expression_vect,
                args,
            );
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
        Strategy::Pulse {
            time_limit,
            phase_limit,
        } => {
            let expression_vect = reader::read_expressions(&args.expressions_file).unwrap();
            let reports = eggshell::baseline::pulse_simplify_expressions::<Halide, _>(
                &AstSize,
                &expression_vect,
                args,
                Some(Duration::from_secs_f64(*time_limit)),
                Some(*phase_limit),
            );
            writer::write_results_csv("tmp/results_simplify.csv", &reports).unwrap();
        }
    }
}
