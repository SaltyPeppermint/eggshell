use std::fmt::Display;

use egg::{Analysis, Language, Pattern, Runner, Searcher};
use log::info;

fn run_check_iter<L, N>(runner: &mut Runner<L, N>, goals: &[Pattern<L>]) -> Result<(), String>
where
    L: Language + Display,
    N: Analysis<L>,
{
    let start_id = runner.egraph.find(*runner.roots.last().unwrap());
    for goal in goals {
        if goal.search_eclass(&runner.egraph, start_id).is_some() {
            info!("Stopping goal {} matched", goal);
            let err_string = format!("Goal {goal} Matched");
            return Err(err_string);
        }
    }
    Ok(())
}

pub(crate) fn run_check_iter_hook<L, N>(
    goals: &[Pattern<L>],
) -> impl FnMut(&mut Runner<L, N>) -> Result<(), String> + 'static
where
    L: Language + Display + 'static,
    N: Analysis<L>,
{
    let goals_vec = goals.to_owned();
    move |runner: &mut Runner<L, N>| run_check_iter(runner, &goals_vec)
}
