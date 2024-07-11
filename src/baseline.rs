use std::time::Duration;

use egg::{CostFunction, Extractor, RecExpr};
use serde::Serialize;

use crate::argparse::EqsatArgs;
use crate::eqsat::results::EqsatResult;
use crate::eqsat::Eqsat;
use crate::io::structs::{EqsatReport, Expression};
use crate::trs::Trs;

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn prove_expr<R, C>(
    expr: &Expression,
    eqsat_args: &EqsatArgs,
    iter_check: bool,
) -> EqsatReport<R::Language, R::Analysis, C>
where
    R: Trs,
    C: CostFunction<R::Language>,
    C::Cost: Serialize,
{
    // For each expression try to prove it then push the results into the results vector.
    let prove_pattern = R::prove_goals();

    println!("Starting Expression: {}", expr.index);
    let start_expr = &expr.term.parse().unwrap();
    let mut eqsat: Eqsat<R> = Eqsat::new(expr.index).with_runner_args(eqsat_args.into());
    let eqsat_result = eqsat.run_goal_once(start_expr, &prove_pattern);

    make_report(
        start_expr.clone(),
        &eqsat,
        expr,
        Some(iter_check),
        eqsat_args,
        eqsat_result,
        vec![],
    )
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn pulse_prove_expr<R, C>(
    expr: &Expression,
    time_limit: Option<Duration>,
    phase_limit: Option<usize>,
    eqsat_args: &EqsatArgs,
    cost_fn: &C,
    iter_check: bool,
) -> EqsatReport<R::Language, R::Analysis, C>
where
    R: Trs,
    C: CostFunction<R::Language> + Clone,
    C::Cost: Serialize,
{
    println!("Starting Expression: {}", expr.index);
    let prove_pattern = R::prove_goals();

    let first_expr = expr.term.parse::<RecExpr<R::Language>>().unwrap();
    let mut start_expr = first_expr.clone();
    let mut eqsat: Eqsat<R> = Eqsat::new(expr.index)
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

        let extractor = Extractor::new(eqsat.last_egraph().unwrap(), cost_fn.clone());
        let extracted: Vec<_> = eqsat
            .last_roots()
            .unwrap()
            .iter()
            .map(|root| extractor.find_best(*root))
            .collect();

        start_expr = extracted.last().unwrap().1.clone();
        extracted_expr.push(extracted);
    };

    make_report(
        first_expr,
        &eqsat,
        expr,
        Some(iter_check),
        eqsat_args,
        final_result,
        extracted_expr,
    )
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn simplify_expr<R, C>(
    expr: &Expression,
    eqsat_args: &EqsatArgs,
    cost_fn: &C,
) -> EqsatReport<R::Language, R::Analysis, C>
where
    R: Trs,
    C: CostFunction<R::Language> + Clone,
    C::Cost: Serialize,
{
    println!("Starting Expression: {}", expr.index);
    let start_expr = &expr.term.parse().unwrap();
    let mut eqsat: Eqsat<R> = Eqsat::new(expr.index).with_runner_args(eqsat_args.into());
    let processed_egraph = eqsat.run_simplify_once(start_expr);

    let extractor = Extractor::new(&processed_egraph, cost_fn.clone());
    let root = eqsat.last_single_root().unwrap();
    let extracted = vec![extractor.find_best(*root)];
    let final_result = EqsatResult::LimitReached(Box::new(processed_egraph));

    make_report(
        start_expr.clone(),
        &eqsat,
        expr,
        None,
        eqsat_args,
        final_result,
        vec![extracted],
    )
}

#[allow(clippy::missing_panics_doc)]
#[must_use]
pub fn pulse_simplify_expr<R, C>(
    expr: &Expression,
    time_limit: Option<Duration>,
    phase_limit: Option<usize>,
    eqsat_args: &EqsatArgs,
    cost_fn: &C,
) -> EqsatReport<<R as Trs>::Language, <R as Trs>::Analysis, C>
where
    R: Trs,
    C: CostFunction<R::Language> + Clone,
    C::Cost: Serialize,
{
    println!("Starting Expression: {}", expr.index);
    let first_expr: RecExpr<R::Language> = expr.term.parse().unwrap();
    let mut start_expr = first_expr.clone();
    let mut eqsat: Eqsat<R> = Eqsat::new(expr.index)
        .with_time_limit(time_limit)
        .with_phase_limit(phase_limit)
        .with_runner_args(eqsat_args.into());

    let mut extracted_expr = Vec::new();

    let egraph = loop {
        let processed_egraph = eqsat.run_simplify_once(&start_expr);
        if eqsat.limit_reached() {
            break processed_egraph;
        }

        let extractor = Extractor::new(&processed_egraph, cost_fn.clone());
        let root = eqsat.last_single_root().unwrap();
        let extracted = vec![extractor.find_best(*root)];

        start_expr = extracted.last().unwrap().1.clone();
        extracted_expr.push(extracted);
    };

    let final_result = EqsatResult::LimitReached(Box::new(egraph));

    make_report(
        first_expr,
        &eqsat,
        expr,
        None,
        eqsat_args,
        final_result,
        extracted_expr,
    )
}

#[allow(clippy::type_complexity)]
fn make_report<R, C>(
    first_expr: RecExpr<R::Language>,
    eqsat: &Eqsat<R>,
    expr: &Expression,
    iter_check: Option<bool>,
    eqsat_args: &EqsatArgs,
    final_result: EqsatResult<R::Language, R::Analysis>,
    extracted_exprs: Vec<Vec<(C::Cost, RecExpr<R::Language>)>>,
) -> EqsatReport<R::Language, R::Analysis, C>
where
    R: Trs,
    C: CostFunction<R::Language>,
    C::Cost: Serialize,
{
    let stats_history = eqsat.stats_history().to_vec();
    EqsatReport {
        index: expr.index,
        first_expr,
        stats_history,
        iteration_check: iter_check,
        total_time: eqsat.total_time(),
        runner_args: eqsat_args.into(),
        other_solver_data: expr.other_solver.clone(),
        extracted_exprs,
        final_result,
    }
}
