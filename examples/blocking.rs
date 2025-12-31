use std::time::Duration;

use egg::{AstSize, RecExpr, Runner, SimpleScheduler};

use eggshell::eqsat::hooks;
use eggshell::rewrite_system::rise::{self, BLOCKING_GOAL, MM, Rise, Ruleset, SPLIT_GUIDE};
use eggshell::sketch;
use eggshell::utils;

fn main() {
    let mm: RecExpr<Rise> = rise::canon_nat(&MM.parse().unwrap());
    let split_guide: RecExpr<Rise> = rise::canon_nat(&SPLIT_GUIDE.parse().unwrap());
    let blocking_goal: RecExpr<Rise> = rise::canon_nat(&BLOCKING_GOAL.parse().unwrap());

    let runner_1 = Runner::default()
        .with_expr(&mm)
        .with_time_limit(Duration::from_secs(60))
        .with_node_limit(10_000_000)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(split_guide.clone()))
        // .with_hook(hooks::printer_hook)
        .run(&rise::rules(Ruleset::Split));

    // println!("{}\nSPLIT STEP DONE DONE\n-----------\n", runner_1.report());
    assert_eq!(
        runner_1.roots[0],
        runner_1.egraph.lookup_expr(&split_guide).unwrap()
    );

    let runner_2 = Runner::default()
        .with_expr(&split_guide)
        .with_time_limit(Duration::from_secs(60))
        .with_node_limit(10_000_000)
        .with_iter_limit(11)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(blocking_goal.clone()))
        .with_hook(hooks::printer_hook)
        .run(&rise::rules(Ruleset::Reorder));

    println!("{}\nREORDER STEP DONE\n-----------\n", runner_2.report());

    let root_guide = runner_2.egraph.find(runner_2.roots[0]);
    let blocking_goal_sketch = rise::sketchify(&blocking_goal, &|n| matches!(n, Rise::NatMul(_)));
    let sketch_extracted_blocking_goal = rise::canon_nat(
        &sketch::eclass_extract(&blocking_goal_sketch, AstSize, &runner_2.egraph, root_guide)
            .unwrap()
            .1,
    );
    assert!(utils::find_diff(&sketch_extracted_blocking_goal, &blocking_goal).is_none());
    assert_eq!(
        runner_2.roots[0],
        runner_2.egraph.lookup_expr(&blocking_goal).unwrap()
    );
}
