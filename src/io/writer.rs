use std::fs::File;
use std::io::BufWriter;

use csv::Writer;
use serde::Serialize;

use crate::errors::EggShellError;

/// Writes the results (a vector of [`Serialize`] items, most likeley
/// [`EqsatResult`]) into a CSV file.
///
/// [`EqsatResult`]: `crate::eqsat::results::EqsatResult`
#[allow(clippy::missing_errors_doc)]
pub fn write_results_csv<T: Serialize>(path: &str, results: &[T]) -> Result<(), EggShellError> {
    let mut wtr = Writer::from_path(path)?;

    for result in results {
        wtr.serialize(result)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Writes the results (a vector of [`Serialize`] items, most likeley
/// [`EqsatResult`]) into a JSON file.
///
/// [`EqsatResult`]: `crate::eqsat::results::EqsatResult`
#[allow(clippy::missing_errors_doc)]
pub fn write_results_json<T: Serialize>(path: &str, results: &[T]) -> Result<(), EggShellError> {
    let file = File::open(path)?;
    let buf_writer = BufWriter::new(file);
    serde_json::to_writer(buf_writer, results)?;
    Ok(())
}
