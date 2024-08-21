#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::redundant_closure_for_method_calls,
    clippy::module_name_repetitions
)]
#![warn(
    clippy::dbg_macro,
    clippy::empty_structs_with_brackets,
    clippy::get_unwrap,
    clippy::map_err_ignore,
    clippy::needless_raw_strings,
    clippy::pub_without_shorthand,
    clippy::redundant_type_annotations,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::str_to_string,
    clippy::string_to_string,
    clippy::string_add,
    absolute_paths_not_starting_with_crate,
    clippy::create_dir,
    clippy::deref_by_slicing,
    clippy::filetype_is_file,
    clippy::format_push_string,
    clippy::impl_trait_in_params,
    clippy::map_err_ignore,
    clippy::pub_without_shorthand,
    clippy::semicolon_inside_block,
    clippy::tests_outside_test_module,
    clippy::todo,
    clippy::unnecessary_safety_comment,
    clippy::unnecessary_safety_doc,
    clippy::unnecessary_self_imports,
    clippy::verbose_file_reads,
    clippy::shadow_unrelated
)]

// clippy::cfg_not_test,
// clippy::use_debug
// clippy::ref_patterns,

pub mod eqsat;
mod io;
mod python;
pub mod sampling;
pub mod sketch;
pub mod trs;
pub mod utils;

use pyo3::prelude::*;

type HashMap<K, V> = hashbrown::HashMap<K, V>;
type HashSet<T> = hashbrown::HashSet<T>;
// type HashMap<K, V> = rustc_hash::crate::HashMap<K, V>;
// type HashSet<K, V> = rustc_hash::crate::HashSet<K, V>;

/// A Python module implemented in Rust.
#[allow(clippy::missing_errors_doc, clippy::shadow_reuse)]
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // IO
    let io_m = PyModule::new_bound(m.py(), "io")?;
    io_m.add_function(wrap_pyfunction!(io::reader::read_exprs, m)?)?;
    io_m.add_class::<io::structs::Expression>()?;
    m.add_submodule(&io_m)?;

    // Eqsat
    let eqsat_m = PyModule::new_bound(m.py(), "eqsat")?;

    // Arithmatic specific generated eqsats
    let arithmatic_m = PyModule::new_bound(eqsat_m.py(), "arithmatic")?;
    arithmatic_m.add_class::<python::arithmatic::Eqsat>()?;
    arithmatic_m.add_class::<python::arithmatic::EqsatResult>()?;
    eqsat_m.add_submodule(&arithmatic_m)?;

    // Halide specific generated eqsats
    let halide_m = PyModule::new_bound(eqsat_m.py(), "halide")?;
    halide_m.add_class::<python::halide::Eqsat>()?;
    halide_m.add_class::<python::halide::EqsatResult>()?;
    eqsat_m.add_submodule(&halide_m)?;

    // SimpleLang specific generated eqsats
    let simple_m = PyModule::new_bound(eqsat_m.py(), "simple")?;
    simple_m.add_class::<python::simple::Eqsat>()?;
    simple_m.add_class::<python::simple::EqsatResult>()?;
    eqsat_m.add_submodule(&simple_m)?;

    // Pylang works for all langs that implement Display
    m.add_class::<python::PyLang>()?;
    m.add_class::<python::PySketch>()?;

    Ok(())
}
