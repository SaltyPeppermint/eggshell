use std::fmt::{Debug, Display};

use egg::{Analysis, AstSize, EClass, EGraph, Id, Language};
use hashbrown::HashMap;
use log::{debug, info};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::sampling::choices::PartialRecExpr;

use super::Sampler;

#[derive(Debug)]
pub struct Greedy<'a, L, N>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    egraph: &'a EGraph<L, N>,
    min_ast_sizes: HashMap<Id, usize>,
}

impl<'a, L, N> Greedy<'a, L, N>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    /// Creates a new [`Greedy<L, N>`].
    ///
    /// Terms picked randomly from the viable ones
    pub fn new(egraph: &'a EGraph<L, N>) -> Self {
        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Sampler ready to start sampling!");
        Greedy {
            egraph,
            min_ast_sizes: ast_sizes,
        }
    }
}

impl<'a, L, N> Sampler<'a, L, N> for Greedy<'a, L, N>
where
    L: Language + Display + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        size_limit: usize,
        partial_rec_expr: &PartialRecExpr<L>,
    ) -> &'c L {
        let min_to_fill_other_open = partial_rec_expr
            .other_open_slots()
            .map(|id| self.min_ast_sizes[&id])
            .sum::<usize>();
        debug!("Required to fill rest: {min_to_fill_other_open}");
        // Budget for the children is one less because the node itself has size one at least
        // We need to substract the minimum to fill the rest of the open positions in the partial AST
        // so we dont run into a situation where we cant finish the AST
        let budget = size_limit
            .checked_sub(partial_rec_expr.n_chosen() + min_to_fill_other_open)
            .unwrap();
        debug!("Budget for this node: {min_to_fill_other_open}");

        // We filter out all the enodes that are too big for our maximum budget
        // (strictly smaller since we need to account for the new node itself)
        // and then pick randomly from those
        eclass
            .nodes
            .iter()
            .filter(|node| {
                let cost = node
                    .children()
                    .iter()
                    .map(|child_id| self.min_ast_sizes[child_id])
                    .sum::<usize>();
                cost < budget
            })
            .choose(rng)
            .expect("EClass cannot be empty")
    }

    fn extractable(&self, id: Id, size_limit: usize) -> bool {
        let canonical_id = self.egraph.find(id);
        self.min_ast_sizes[&canonical_id] <= size_limit
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }
}
