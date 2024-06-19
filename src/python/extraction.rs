use hashbrown::HashMap as HashBrownMap;
use pyo3::prelude::*;

use crate::cost_fn;
use crate::errors::EggShellError;
use crate::extraction;
use crate::extraction::Extractor;

use super::eqsat::PyEqsatHalide;

/// Extracts the best term based on the values given in the hashmap.
/// The key is the index in the vector of nodes.
///
/// # Errors
///
/// Will return [`EggShellError::MissingEqsat`] if no equality saturation has
/// previously been run
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn extract_with_costs_bottom_up(
    eqsat: PyEqsatHalide,
    node_costs: HashBrownMap<usize, f64>,
) -> Result<HashBrownMap<String, f64>, EggShellError> {
    let class_specific_costs = eqsat.eqsat.remap_costs(&node_costs)?;
    let cost_fn = cost_fn::LookupCost::new(class_specific_costs);

    let last_egraph = eqsat
        .eqsat
        .last_egraph()
        .ok_or(EggShellError::MissingEqsat)?;
    let last_roots = eqsat
        .eqsat
        .last_roots()
        .ok_or(EggShellError::MissingEqsat)?;
    let extractor = extraction::BottomUp::new(cost_fn, last_egraph);
    let extracted = extractor.extract(last_roots);

    let stringified = extracted
        .iter()
        .map(|x| (x.expr.to_string(), x.cost))
        .collect();
    Ok(stringified)
}

/// Extracts the best term based on the term size
///
/// # Errors
///
/// Will return [`EggShellError::MissingEqsat`] if no equality saturation has
/// previously been run.
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn extract_ast_size_bottom_up(
    eqsat: PyEqsatHalide,
) -> Result<HashBrownMap<String, f64>, EggShellError> {
    let cost_fn = cost_fn::ExprSize;

    let last_egraph = eqsat
        .eqsat
        .last_egraph()
        .ok_or(EggShellError::MissingEqsat)?;
    let last_roots = eqsat
        .eqsat
        .last_roots()
        .ok_or(EggShellError::MissingEqsat)?;
    let extractor = extraction::BottomUp::new(cost_fn, last_egraph);
    let extracted = extractor.extract(last_roots);

    let stringified = extracted
        .iter()
        .map(|x| (x.expr.to_string(), x.cost))
        .collect();
    Ok(stringified)
}
