use egg::{Analysis, EGraph, Language, Rewrite, RewriteScheduler, SearchMatches, Symbol};
use hashbrown::HashMap;

/// A [`BudgetScheduler`] that implements a simple stack of rules to apply.
/// Once the stack has been used up, it becomes the [`egg::SimpleScheduler`].
#[derive(Debug)]
pub struct BudgetScheduler(HashMap<Symbol, usize>);

impl BudgetScheduler {
    /// Build a new Stack Scheduler from a sequence of rewrites
    #[must_use]
    pub fn new(budget: HashMap<Symbol, usize>) -> Self {
        Self(budget)
    }

    #[must_use]
    pub fn from_expl(mut names: Vec<Symbol>) -> Self {
        names.reverse();
        let budget = names
            .into_iter()
            .fold(HashMap::<Symbol, usize>::new(), |mut acc, name| {
                acc.entry(name).and_modify(|x| *x += 1).or_insert(1);
                acc
            });
        Self(budget)
    }
}

impl Default for BudgetScheduler {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<L, N> RewriteScheduler<L, N> for BudgetScheduler
where
    L: Language,
    N: Analysis<L>,
{
    fn can_stop(&mut self, _iteration: usize) -> bool {
        self.0.is_empty()
    }

    fn search_rewrite<'a>(
        &mut self,
        _iteration: usize,
        egraph: &EGraph<L, N>,
        rewrite: &'a Rewrite<L, N>,
    ) -> Vec<SearchMatches<'a, L>> {
        if self.0.is_empty() {
            // Fall back to greedy schedule once the budget has been used up
            rewrite.search(egraph)
        } else if let Some(v) = self.0.get_mut(&rewrite.name)
            && *v != 0
        {
            // Take one out of the budget and apply the rewrites
            *v -= 1;
            rewrite.search(egraph)
        } else {
            // Skip rewrites that are not in the budget
            vec![]
        }
    }
}
