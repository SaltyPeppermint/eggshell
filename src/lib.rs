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
    clippy::shadow_unrelated,
    // clippy::use_debug,
    // clippy::cfg_not_test,
    // clippy::allow_attributes
)]

mod analysis;
mod error;
mod features;
pub mod python;
mod utils;

pub mod cli;
pub mod eqsat;
pub mod io;
pub mod sampling;
pub mod sketch;
pub mod trs;
pub mod typing;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    // m.add_class::<python:>()?;
    python::simple::add_mod(m, "simple")?;
    python::arithmetic::add_mod(m, "arithmetic")?;
    python::halide::add_mod(m, "halide")?;
    python::rise::add_mod(m, "rise")?;

    m.add(
        "EggshellException",
        m.py().get_type::<error::EggshellException>(),
    )?;

    Ok(())
}
