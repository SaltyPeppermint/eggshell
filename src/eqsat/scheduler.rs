use egg::{Analysis, EGraph, Language, Rewrite, RewriteScheduler, SearchMatches, Symbol};

/// A [`StackScheduler`] that implements a simple stack of rules to apply.
/// Once the stack has been used up, it becomes the [`egg::SimpleScheduler`].
#[derive(Debug)]
pub struct StackScheduler(Vec<Symbol>);

impl StackScheduler {
    /// Build a new Stack Scheduler from a sequence of rewrites
    pub fn new(mut names: Vec<Symbol>) -> StackScheduler {
        names.reverse();
        Self(names)
    }
}

impl Default for StackScheduler {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<L, N> RewriteScheduler<L, N> for StackScheduler
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
        if self.0.first().map(|n| n == &rewrite.name).unwrap_or(true) {
            self.0.pop();
            rewrite.search(egraph)
        } else {
            vec![]
        }
    }
}
