use std::fs::File;

use csv;
use csv::StringRecord;
use pyo3::prelude::*;

use super::structs::{Expression, OtherSolverData};
use crate::errors::EggShellError;

/// Reads expressions from a csv file into a vector of [`Expression`] Vector.
///
/// [`Expression`]: super::structs::Expression
#[allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
#[pyfunction]
pub fn read_exprs(file_path: &str) -> Result<Vec<Expression>, EggShellError> {
    // Declare the vector and the reader
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    // Read each record and extract then cast the values.
    rdr.records()
        .map(|result| {
            result.map_err(std::convert::Into::into).and_then(|record| {
                let index = record
                    .get(0)
                    .unwrap()
                    .parse::<usize>()
                    .expect("No index means csv is broken.");
                let expression = record.get(1).expect("No expression means csv is broken.");
                // Check if Halide's resluts are included then add them if they are
                let other_solver_data = parse_other_solver(&record)?;

                // Push the new ExpressionStruct initialized with the values extracted into the vector.
                Ok(Expression {
                    index,
                    term: (*expression).to_string(),
                    other_solver: other_solver_data,
                })
            })
        })
        .collect()
}

fn parse_other_solver(record: &StringRecord) -> Result<Option<OtherSolverData>, EggShellError> {
    if let (Some(halide_result), Some(other_time)) = (record.get(2), record.get(3)) {
        let other_time = other_time.parse()?;
        Ok(Some(OtherSolverData {
            result: halide_result.into(),
            time: other_time,
        }))
    } else {
        Ok(None)
    }
}
