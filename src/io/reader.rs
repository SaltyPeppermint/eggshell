use std::fs::File;
use std::path::Path;

use csv;
use serde::{Deserialize, Serialize};

// use pyo3::prelude::*;

use super::structs::Entry;

/// Reads expressions from a csv file into a vector of [`Expression`] Vector.
///
/// [`Expression`]: super::structs::Expression
#[expect(clippy::missing_panics_doc)]
#[must_use]
pub fn read_exprs_csv(file_path: &Path) -> Vec<Entry> {
    // Declare the vector and the reader
    let file = File::open(file_path).expect("CSV File needs to exist");
    let mut rdr = csv::Reader::from_reader(file);
    // Read each record and extract then cast the values.
    rdr.records()
        .map(|result| {
            let record = result.expect("CSV must be properly formatted");
            let index = record
                .get(0)
                .expect("No index means csv is broken.")
                .parse::<usize>()
                .expect("Wrong index means csv is broken.");
            let expr_str = record.get(1).expect("No expression means csv is broken.");
            let truth_value = record.get(2).map(|s| s.to_owned());

            // Push the new ExpressionStruct initialized with the values extracted into the vector.
            Entry {
                index,
                expr: expr_str.to_owned(),
                truth_value,
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct JsonExpression {
    start: String,
    end: String,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct JsonDatapoint {
    expression: JsonExpression,
    rules: Vec<String>,
}

#[expect(clippy::missing_panics_doc)]
#[must_use]
pub fn read_exprs_json(file_path: &Path) -> Vec<Entry> {
    // Declare the vector and the reader
    let file = File::open(file_path).expect("Json file must exist");
    let data: Vec<JsonDatapoint> =
        serde_json::from_reader(file).expect("Needs to be properly formatted json file");
    // Read each record and extract then cast the values.
    data.into_iter()
        .enumerate()
        .map(|(index, entry)| Entry {
            index,
            expr: entry.expression.start,
            truth_value: Some(entry.expression.end),
        })
        .collect()
}
