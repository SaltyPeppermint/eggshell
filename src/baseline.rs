use std::time::Duration;

use egg::RecExpr;

use crate::argparse::EqsatArgs;
use crate::cost_fn::ExprSize;
use crate::eqsat::results::EqsatResult;
use crate::eqsat::Eqsat;
use crate::extraction::{BottomUp, ExtractResult, Extractor};
use crate::flattened::FlatGraph;
use crate::io::structs::{EqsatReport, Expression};
use crate::trs::Trs;

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn prove_expressions<R>(
    exprs_vect: &[Expression],
    eqsat_args: &EqsatArgs,
    iter_check: bool,
) -> Vec<EqsatReport<R::Language, String, FlatGraph>>
where
    R: Trs,
{
    // For each expression try to prove it then push the results into the results vector.
    let prove_pattern = R::prove_goals();

    exprs_vect
        .iter()
        .map(|expression| {
            println!("Starting Expression: {}", expression.index);
            let start_expr = &expression.term.parse().unwrap();
            let mut eqsat: Eqsat<R> = Eqsat::new(expression.index)
                .unwrap()
                .with_iteration_check(iter_check)
                .with_runner_args(eqsat_args.into());
            let eqsat_result = eqsat.run_goal_once(start_expr, &prove_pattern);

            make_report(
                start_expr.clone(),
                &eqsat,
                expression,
                Some(iter_check),
                eqsat_args,
                eqsat_result,
                vec![],
            )
        })
        .collect()
}

/// Using multiple phases, no forking, only ever extracting a single expression
#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn pulse_prove_expressions<R>(
    exprs_vect: &[Expression],
    eqsat_args: &EqsatArgs,
    time_limit: Option<Duration>,
    phase_limit: Option<usize>,
    iter_check: bool,
) -> Vec<EqsatReport<R::Language, String, FlatGraph>>
where
    R: Trs,
{
    // For each expression try to prove it then push the results into the results vector.
    let prove_pattern = R::prove_goals();

    exprs_vect
        .iter()
        .map(|expression| {
            println!("Starting Expression: {}", expression.index);
            let first_expr: RecExpr<R::Language> = expression.term.parse().unwrap();
            let mut start_expr = first_expr.clone();
            let mut eqsat: Eqsat<R> = Eqsat::new(expression.index)
                .unwrap()
                .with_iteration_check(iter_check)
                .with_time_limit(time_limit)
                .with_phase_limit(phase_limit)
                .with_runner_args(eqsat_args.into());

            let mut extracted_expr = Vec::new();

            let final_result = loop {
                let eqsat_result = eqsat.run_goal_once(&start_expr, &prove_pattern);

                if eqsat.limit_reached() {
                    break eqsat_result;
                }
                if let EqsatResult::Solved(_) | EqsatResult::Undecidable = eqsat_result {
                    break eqsat_result;
                }

                let extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
                let extracted = extractor.extract(eqsat.last_roots().unwrap());

                start_expr = extracted.last().unwrap().expr.clone();
                extracted_expr.push(extracted);
            };

            make_report(
                first_expr,
                &eqsat,
                expression,
                Some(iter_check),
                eqsat_args,
                final_result,
                extracted_expr,
            )
        })
        .collect()
}

/// Runs Simple to simplify the expressions passed as vector using the different params passed.
#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn simplify_expressions<R>(
    exprs_vect: &[Expression],
    eqsat_args: &EqsatArgs,
) -> Vec<EqsatReport<R::Language, String, FlatGraph>>
where
    R: Trs,
{
    exprs_vect
        .iter()
        .map(|expression| {
            println!("Starting Expression: {}", expression.index);
            let start_expr = &expression.term.parse().unwrap();
            let mut eqsat: Eqsat<R> = Eqsat::new(expression.index)
                .unwrap()
                .with_runner_args(eqsat_args.into());
            let flat_graph = eqsat.run_simplify_once(start_expr);
            let extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
            let extracted = extractor.extract(eqsat.last_roots().unwrap());

            make_report(
                start_expr.clone(),
                &eqsat,
                expression,
                None,
                eqsat_args,
                EqsatResult::LimitReached(flat_graph),
                vec![extracted],
            )
        })
        .collect()
}

/// Using multiple phases, no forking, only ever extracting a single expression
#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn pulse_simplify_expressions<R>(
    exprs_vect: &[Expression],
    eqsat_args: &EqsatArgs,
    time_limit: Option<Duration>,
    phase_limit: Option<usize>,
) -> Vec<EqsatReport<R::Language, String, FlatGraph>>
where
    R: Trs,
{
    // For each expression try to prove it then push the results into the results vector.
    exprs_vect
        .iter()
        .map(|expression| {
            println!("Starting Expression: {}", expression.index);
            let first_expr: RecExpr<R::Language> = expression.term.parse().unwrap();
            let mut start_expr = first_expr.clone();
            let mut eqsat: Eqsat<R> = Eqsat::new(expression.index)
                .unwrap()
                .with_time_limit(time_limit)
                .with_phase_limit(phase_limit)
                .with_runner_args(eqsat_args.into());

            let mut extracted_expr = Vec::new();

            let flat_graph = loop {
                let eqsat_result = eqsat.run_simplify_once(&start_expr);

                if eqsat.limit_reached() {
                    break eqsat_result;
                }

                let extractor = BottomUp::new(ExprSize, eqsat.last_egraph().unwrap());
                let extracted = extractor.extract(eqsat.last_roots().unwrap());

                start_expr = extracted.last().unwrap().expr.clone();
                extracted_expr.push(extracted);
            };

            make_report(
                first_expr,
                &eqsat,
                expression,
                None,
                eqsat_args,
                EqsatResult::LimitReached(flat_graph),
                extracted_expr,
            )
        })
        .collect()
}

fn make_report<R, T, U>(
    first_expr: RecExpr<R::Language>,
    eqsat: &Eqsat<R>,
    expression: &Expression,
    iteration_check: Option<bool>,
    eqsat_args: &EqsatArgs,
    final_result: EqsatResult<T, U>,
    extracted_exprs: Vec<Vec<ExtractResult<R::Language>>>,
) -> EqsatReport<R::Language, T, U>
where
    R: Trs,
{
    let stats_history = eqsat.stats_history().to_vec();
    EqsatReport {
        index: expression.index,
        first_expr,
        stats_history,
        iteration_check,
        total_time: eqsat.total_time(),
        runner_args: eqsat_args.into(),
        other_solver_data: expression.other_solver.clone(),
        extracted_exprs,
        final_result,
    }
}
