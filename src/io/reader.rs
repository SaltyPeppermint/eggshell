use std::fs::File;

use csv;
use pyo3::prelude::*;

use super::structs::Expression;
use super::IoError;

/// Reads expressions from a csv file into a vector of [`Expression`] Vector.
///
/// [`Expression`]: super::structs::Expression
#[allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
#[pyfunction]
pub fn read_exprs(file_path: &str) -> Result<Vec<Expression>, IoError> {
    // Declare the vector and the reader
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    // Read each record and extract then cast the values.
    rdr.records()
        .map(|result| {
            let record = result?;
            let index = record
                .get(0)
                .expect("No index means csv is broken.")
                .parse::<usize>()
                .expect("Wrong index means csv is broken.");
            let expr_str = record.get(1).expect("No expression means csv is broken.");

            // Push the new ExpressionStruct initialized with the values extracted into the vector.
            Ok(Expression {
                index,
                term: (*expr_str).to_string(),
            })
        })
        .collect()
}
