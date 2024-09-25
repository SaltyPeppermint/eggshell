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
    clippy::allow_attributes
)]

// clippy::cfg_not_test,
// clippy::use_debug
// clippy::ref_patterns,

pub mod eqsat;
pub mod io;
pub(crate) mod python;
pub mod sampling;
pub mod sketch;
pub mod trs;
pub mod typing;
pub mod utils;

use pyo3::prelude::*;

// type HashMap<K, V> = hashbrown::HashMap<K, V>;
// type HashSet<T> = hashbrown::HashSet<T>;
// type HashMap<K, V> = rustc_hash::crate::HashMap<K, V>;
// type HashSet<K, V> = rustc_hash::crate::HashSet<K, V>;

/// A Python module implemented in Rust.
#[expect(clippy::shadow_reuse)]
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    m.add_class::<python::PyLang>()?;
    m.add_class::<python::PySketch>()?;
    m.add_class::<python::FlatAst>()?;
    m.add_class::<python::FlatNode>()?;
    m.add_class::<python::FlatEGraph>()?;
    m.add_class::<python::FlatVertex>()?;
    m.add_function(wrap_pyfunction!(python::sketch_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(python::open_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(python::active_symbol, m)?)?;

    python::simple::add_mod(m, "simple")?;
    python::arithmatic::add_mod(m, "arithmatic")?;
    python::halide::add_mod(m, "halide")?;

    Ok(())
}
