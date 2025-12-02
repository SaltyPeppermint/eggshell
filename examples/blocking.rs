use std::time::Duration;

use egg::{AstSize, RecExpr, Runner, SimpleScheduler};

use eggshell::eqsat::hooks;
use eggshell::rewrite_system::rise::{self, Rise, RiseRuleset};
use eggshell::rewrite_system::rise::{BLOCKING_GOAL, MM, SPLIT_GUIDE};
use eggshell::sketch;

fn main() {
    let mm: RecExpr<Rise> = MM.parse().unwrap();
    let split_guide: RecExpr<Rise> = SPLIT_GUIDE.parse().unwrap();

    let runner_1 = Runner::default()
        .with_expr(&mm)
        .with_iter_limit(5)
        .with_time_limit(Duration::from_secs(30))
        .with_node_limit(1_000_000)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(split_guide.clone()))
        .run(&rise::rules(RiseRuleset::MM));

    println!("{}", runner_1.report());

    let root_mm = runner_1.egraph.find(runner_1.roots[0]);
    let baseline_sketch = sketch::sketchify(SPLIT_GUIDE);
    let (_, sketch_extracted_split_guide) =
        sketch::eclass_extract(&baseline_sketch, AstSize, &runner_1.egraph, root_mm).unwrap();

    assert_eq!(
        None,
        eggshell::utils::find_diff(&sketch_extracted_split_guide, &split_guide)
    );
    assert_eq!(root_mm, runner_1.egraph.lookup_expr(&split_guide).unwrap());

    let blocking_goal: RecExpr<Rise> = BLOCKING_GOAL.parse().unwrap();
    let runner_2 = Runner::default()
        .with_expr(&split_guide)
        .with_iter_limit(6)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(blocking_goal.clone()))
        .run(&rise::rules(RiseRuleset::MM));

    let root_guide = runner_2.egraph.find(runner_2.roots[0]);
    assert_eq!(
        root_guide,
        runner_2.egraph.lookup_expr(&blocking_goal).unwrap()
    );
}
