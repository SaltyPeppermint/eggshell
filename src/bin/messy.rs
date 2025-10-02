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

use egg::{AstSize, CostFunction, RecExpr, SimpleScheduler};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use eggshell::eqsat::{self, EqsatConf};
use eggshell::rewrite_system::rise::RiseLang;
use eggshell::rewrite_system::{RewriteSystem, Rise};
use eggshell::sampling::sampler::{Greedy, Sampler};

fn main() {
    let start_expr ="(lam f1 (lam f2 (lam f3 (lam f4 (lam f5 (lam x3 (app (app map (var f5)) (app (lam x2 (app (app map (var f4)) (app (lam x1 (app (app map (var f3)) (app (lam x0 (app (app map (var f2)) (app (app map (var f1)) (var x0)))) (var x1)))) (var x2)))) (var x3)))))))))".parse::<RecExpr<RiseLang>>().unwrap();

    let rules = Rise::full_rules();

    let mut penultimate_result = eqsat::eqsat(
        EqsatConf::builder()
            .node_limit(100_000_000_000)
            .iter_limit(6)
            .build(),
        (&start_expr).into(),
        &rules,
        None,
        SimpleScheduler,
    );

    let final_result = eqsat::eqsat(
        EqsatConf::builder()
            .node_limit(100_000_000_000)
            .iter_limit(7)
            .build(),
        (&start_expr).into(),
        &rules,
        None,
        SimpleScheduler,
    );

    println!("Stop reason: {:?}", final_result.report().stop_reason);

    let min_size = AstSize.cost_rec(&start_expr);
    let max_size = min_size * 2;
    let rng = ChaCha12Rng::seed_from_u64(0);
    let samples = Greedy::new(final_result.egraph())
        .sample_eclass(&rng, 100, final_result.roots()[0], max_size, 1)
        .unwrap();

    println!("Printing frontier expr");
    let filtered = samples
        .into_iter()
        .filter_map(|s| {
            penultimate_result
                .egraph_mut()
                .lookup_expr(&s)
                .is_none()
                .then_some(s)
        })
        .collect::<Vec<_>>();
    let output = serde_json::to_string(&filtered).unwrap();
    println!("Number of filtered samples: {}", filtered.len());
    println!("---\n{output}\n---");
}
