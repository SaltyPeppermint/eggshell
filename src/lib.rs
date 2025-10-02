#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::module_name_repetitions)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::empty_structs_with_brackets)]
#![warn(clippy::get_unwrap)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::needless_raw_strings)]
#![warn(clippy::pub_without_shorthand)]
#![warn(clippy::redundant_type_annotations)]
#![warn(clippy::shadow_reuse)]
#![warn(clippy::shadow_same)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::string_add)]
#![warn(clippy::absolute_paths_not_starting_with_crate)]
#![warn(clippy::create_dir)]
#![warn(clippy::deref_by_slicing)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::format_push_string)]
#![warn(clippy::impl_trait_in_params)]
#![warn(clippy::semicolon_inside_block)]
#![warn(clippy::tests_outside_test_module)]
#![warn(clippy::todo)]
#![warn(clippy::unnecessary_safety_comment)]
#![warn(clippy::unnecessary_safety_doc)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::verbose_file_reads)]
#![warn(clippy::shadow_unrelated)]
// Note: use_debug, cfg_not_test, and allow_attributes are not valid Clippy lints
#![warn(clippy::dbg_macro)]

mod analysis;
mod utils; //should maybe be deleted
mod viz;

pub mod cli;
pub mod eqsat;
pub mod explanation;
pub mod io;
pub mod probabilistic;
pub mod python;
pub mod rewrite_system;
pub mod sampling;
pub mod sketch;
pub mod tree_distance;
