use egg::{RecExpr, Runner, SimpleScheduler};

use eggshell::eqsat::hooks;
use eggshell::rise::{self, BASELINE_GOAL, MM, Rise, Ruleset};

fn main() {
    let mm: RecExpr<Rise> = MM.parse().unwrap();
    let baseline_goal: RecExpr<Rise> = BASELINE_GOAL.parse().unwrap();

    let runner = Runner::default()
        .with_expr(&mm)
        .with_iter_limit(3)
        .with_scheduler(SimpleScheduler)
        .with_hook(hooks::targe_hook(baseline_goal.clone()))
        .run(&rise::rules(Ruleset::All));

    let root_mm = runner.egraph.find(runner.roots[0]);

    assert_eq!(root_mm, runner.egraph.lookup_expr(&mm).unwrap());
    assert_eq!(root_mm, runner.egraph.lookup_expr(&baseline_goal).unwrap());
}
