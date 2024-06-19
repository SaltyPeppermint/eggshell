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
pub fn read_expressions(file_path: &str) -> Result<Vec<Expression>, EggShellError> {
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

// /// Reads the expressions in the format specified for the work done for the paper variant.
// #[allow(clippy::missing_errors_doc)]
// fn read_expressions_paper(file_path: &str) -> Result<Vec<(String, String)>, EggShellError> {
//     let mut expressions_vect = Vec::new();
//     let file = File::open(file_path)?;
//     let mut rdr = csv::ReaderBuilder::new().delimiter(b';').from_reader(file);
//     for result in rdr.records() {
//         let record = result?;
//         let infix = record[0].to_string();
//         let prefix = record[1].to_string();
//         expressions_vect.push((infix, prefix));
//     }
//     Ok(expressions_vect)
// }

// /// Reads the rules from a CSV file then pareses them into a Rule Vector.
// #[allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
// fn read_rules(file_path: &str) -> Result<Vec<Rule>, EggShellError> {
//     let mut rules_vect: Vec<Rule> = Vec::new();
//     let file = File::open(file_path)?;
//     let mut rdr = csv::Reader::from_reader(file);
//     for result in rdr.records() {
//         let record = result?;
//         let index = record[0].parse::<usize>().unwrap();
//         let lhs = record[2].to_string();
//         let rhs = record[3].to_string();
//         let condition = record[4].to_string();
//         rules_vect.push(Rule::new(index, lhs, rhs, Some(condition)));
//     }
//     Ok(rules_vect)
// }
