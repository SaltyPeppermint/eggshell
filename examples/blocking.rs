use std::time::Duration;

use egg::{AstSize, Runner, SimpleScheduler};

use eggshell::eqsat::hooks;
use eggshell::rewrite_system::rise::{self, BLOCKING_GOAL, MM, RiseRuleset, SPLIT_GUIDE};
use eggshell::sketch;
use eggshell::utils;

fn main() {
    let mm = rise::canon_nat(&MM.parse().unwrap());
    let split_guide = rise::canon_nat(&SPLIT_GUIDE.parse().unwrap());
    let blocking_goal = rise::canon_nat(&BLOCKING_GOAL.parse().unwrap());

    let runner_1 = Runner::default()
        .with_expr(&mm)
        .with_iter_limit(6)
        .with_time_limit(Duration::from_secs(60))
        .with_node_limit(1_000_000)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(split_guide.clone()))
        .with_hook(hooks::printer_hook)
        .run(&rise::rules(RiseRuleset::MM));

    println!("{}", runner_1.report());

    let root_mm = runner_1.egraph.find(runner_1.roots[0]);
    let split_guide_sketch = rise::sketchify(&split_guide, true);
    let sketch_extracted_split_guide = rise::canon_nat(
        &sketch::eclass_extract(&split_guide_sketch, AstSize, &runner_1.egraph, root_mm)
            .unwrap()
            .1,
    );
    assert!(utils::find_diff(&sketch_extracted_split_guide, &split_guide).is_none());

    // println!("\nGuide Ground Truth");
    // split_guide.pp(false);
    // println!("\nSketch Extracted:");
    // sketch_extracted_split_guide.pp(false);

    // assert_eq!(root_mm, runner_1.egraph.lookup_expr(&split_guide).unwrap());

    let runner_2 = Runner::default()
        .with_expr(&split_guide)
        .with_iter_limit(8)
        .with_time_limit(Duration::from_secs(60))
        .with_node_limit(1_000_000)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(blocking_goal.clone()))
        .with_hook(hooks::printer_hook)
        .run(&rise::rules(RiseRuleset::MM));

    println!("{}", runner_2.report());

    let root_guide = runner_2.egraph.find(runner_2.roots[0]);
    let blocking_goal_sketch = rise::sketchify(&blocking_goal, true);
    let sketch_extracted_blocking_goal = rise::canon_nat(
        &sketch::eclass_extract(&blocking_goal_sketch, AstSize, &runner_2.egraph, root_guide)
            .unwrap()
            .1,
    );
    assert!(utils::find_diff(&sketch_extracted_blocking_goal, &blocking_goal).is_none());
    // assert_eq!(
    //     root_guide,
    //     runner_2.egraph.lookup_expr(&blocking_goal).unwrap()
    // );
}
