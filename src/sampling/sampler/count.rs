use std::fmt::{Debug, Display};

use egg::{Analysis, AstSize, EClass, EGraph, Id, Language};
use hashbrown::HashMap;
use log::{debug, info};
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

use crate::analysis::commutative_semigroup::{CommutativeSemigroupAnalysis, Counter, ExprCount};
use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::sampling::choices::PartialRecExpr;

use super::Sampler;

/// Buggy budget consideration
#[derive(Debug)]
pub struct CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    egraph: &'a EGraph<L, N>,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    // flattened_size_counts: HashMap<Id, C>,
    ast_sizes: HashMap<Id, usize>,
    analysis_depth: usize,
}

impl<'a, C, L, N> CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    /// Creates a new [`CountWeighted<C, L, N>`].
    ///
    /// Terms are weighted according to the number of terms up to the size
    /// cutoff limit
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new(egraph: &'a EGraph<L, N>, analysis_depth: usize) -> Self {
        info!("Using analysis_depth {analysis_depth}");

        // Make one big size count analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        let size_counts = ExprCount::new(analysis_depth).one_shot_analysis(egraph);
        info!("Size count oneshot analysis finsished!");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Sampler ready to start sampling!");
        CountWeightedGreedy {
            egraph,
            size_counts,
            ast_sizes,
            analysis_depth,
        }
    }

    fn pick_by_size_counts<'c>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, <N as Analysis<L>>::Data>,
    ) -> &'c L {
        eclass
            .nodes
            .choose_weighted(rng, |node| {
                node.children()
                    .iter()
                    .map(|child_id| {
                        self.size_counts[child_id]
                            .iter()
                            .map(|(_, count)| count)
                            .sum::<C>()
                    })
                    .product::<C>()
            })
            .or_else(|e| match e {
                // If all weights are zero, we are already way too big and we don't have
                // any data about the options to pick we are reasoning about.
                // We need to pick according to AstSize now to "close out" the expr as fast
                // as possible to prevent it from growing even more.
                WeightedError::AllWeightsZero => {
                    Ok(pick_by_ast_size::<L, N>(&self.ast_sizes, eclass))
                }
                _ => Err(e),
            })
            .expect("NoItem, InvalidWeight and TooMany variants should never trigger.")
    }
}

impl<'a, C, L, N> Sampler<'a, L, N> for CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Display + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        size_limit: usize,
        partial_rec_expr: &PartialRecExpr<L>,
    ) -> &'c L {
        if partial_rec_expr.len() <= size_limit {
            self.pick_by_size_counts(rng, eclass)
        } else {
            pick_by_ast_size::<L, N>(&self.ast_sizes, eclass)
        }
    }

    fn extractable(&self, id: Id, size_limit: usize) -> bool {
        let canonical_id = self.egraph.find(id);
        self.size_counts
            .get(&canonical_id)
            .is_some_and(|eclass_size_counts| {
                eclass_size_counts
                    .iter()
                    .any(|(size, _)| size <= &size_limit)
            })
    }

    fn analysis_depth(&self) -> Option<usize> {
        Some(self.analysis_depth)
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }
}

fn pick_by_ast_size<'a, L: Language, N: Analysis<L>>(
    ast_sizes: &HashMap<Id, usize>,
    eclass: &'a EClass<L, <N as Analysis<L>>::Data>,
) -> &'a L {
    eclass
        .nodes
        .iter()
        .map(|node| {
            let cost = node
                .children()
                .iter()
                .map(|child_id| ast_sizes[child_id])
                .sum::<usize>();
            (node, cost)
        })
        .min_by(|a, b| a.1.cmp(&b.1))
        .expect("EClasses can't have 0 members")
        .0
}

#[derive(Debug)]
pub struct CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    egraph: &'a EGraph<L, N>,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    min_ast_sizes: HashMap<Id, usize>,
    analysis_depth: usize,
}

impl<'a, C, L, N> CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Display + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    /// Creates a new [`CountWeightedUniformly<C, L, N>`].
    ///
    /// Terms are weighted according to the number of terms up to the size
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new(egraph: &'a EGraph<L, N>, analysis_depth: usize) -> Self {
        info!("Using analysis_depth {analysis_depth}");

        // Make one big size count analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        let size_counts = ExprCount::new(analysis_depth).one_shot_analysis(egraph);
        info!("Size count oneshot analysis finsished!");

        let mut min_ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut min_ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Sampler ready to start sampling!");
        CountWeightedUniformly {
            egraph,
            size_counts,
            min_ast_sizes,
            analysis_depth,
        }
    }

    fn calc_node_weight<I: Iterator<Item = Vec<usize>>>(
        &self,
        budget_combinations: I,
        node: &L,
    ) -> C {
        budget_combinations
            // Go over all the possible combinations and zip them with the children of the node
            // Guranteed to be same length
            .map(|budget_combination| {
                node.children()
                    .iter()
                    .zip(budget_combination)
                    // For each child we have a specific budget in this combination
                    .map(|(child_id, child_budget)| {
                        // We only look at the counts that fit into that childs specific budget
                        // for this combination to spend the budget
                        self.size_counts[child_id].get(&child_budget)
                    })
                    // And multiply up the number of terms this specific combination would give us
                    .product::<Option<C>>()
                    // If for a combination any child has no expression, the combination is impossible
                    // and we need to default to 0 as it does not contribute to the count
                    .unwrap_or_else(|| C::zero())
            })
            .sum::<C>()
    }
}

impl<'a, C, L, N> Sampler<'a, L, N> for CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Display + Sync + Send,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        size_limit: usize,
        partial_rec_expr: &PartialRecExpr<L>,
    ) -> &'c L {
        debug!("Current EClass {eclass:?}");
        debug!("Choices: {partial_rec_expr:?}");
        // We need to know what is the minimum size required to fill the rest of the open positions
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

        debug!("Size limit: {size_limit}");
        debug!("Budget available: {budget}");
        debug!("Current EClass Counts {:?}", self.size_counts[&eclass.id]);
        // There has to be at least budget 1 left or something went horribly wrong
        // assert!(budget > 0);

        let weights = eclass
            .nodes
            .iter() // previously par_iter
            .map(|node| {
                debug!("Node being weighted: {node}");
                // Get all the ways we could divide up the budget AND PARTS OF THE BUDGET
                // among the children of the node under consideration
                let arity = node.children().len();
                let budget_combinations = (0..=budget).flat_map(|i| sum_combinations(i, arity));
                let count = self.calc_node_weight(budget_combinations, node);
                (node, count)
            })
            .collect::<HashMap<_, _>>();

        eclass
            .nodes
            .choose_weighted(rng, |node| {
                weights.get(node).expect("Every node has a weight.")
            })
            .expect("NoItem, InvalidWeight and TooMany variants should never trigger.")
    }

    fn extractable(&self, id: Id, size_limit: usize) -> bool {
        let canonical_id = self.egraph.find(id);
        // debug_assert!(canonical_id == id);
        self.size_counts
            .get(&canonical_id)
            .is_some_and(|eclass_size_counts| {
                eclass_size_counts
                    .iter()
                    .any(|(size, _)| size <= &size_limit)
            })
    }

    fn analysis_depth(&self) -> Option<usize> {
        Some(self.analysis_depth)
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }
}

/// Returns the number of ways you can write `num` as the sum of `n` positive natural numbers
fn sum_combinations(num: usize, n: usize) -> Vec<Vec<usize>> {
    fn rec(num: usize, n: usize, current: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if n == 1 {
            // If only one number is left, add the remainder of the sum
            if num >= 1 {
                current.push(num);
                result.push(current.clone());
                current.pop();
            }
            return;
        }

        for i in 1..=num - (n - 1) {
            current.push(i);
            rec(num - i, n - 1, current, result);
            current.pop();
        }
    }

    // How many ways can you write an integer num as the sum of n positive natural numbers?
    // Zero is the only number that can be written as the sum of n natural numbers
    if num == 0 && n == 0 {
        return vec![Vec::new()];
    }
    // If num < n => Not possible => Empty vector
    // if num = 0 => 0 is the sum of no positive integers => Empty vector
    // If n = 0 => No way to write smth other than 0 as the sum of 0 positive numbers => Empty vector
    if num < n || num == 0 || n == 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current = Vec::new();
    rec(num, n, &mut current, &mut result);
    result
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;
    use num::BigUint;
    use rand::SeedableRng;

    use super::*;
    use crate::eqsat::{Eqsat, EqsatConf};
    use crate::rewrite_system::{Halide, RewriteSystem, Simple};
    use crate::sampling::SampleError;

    #[test]
    fn simple_sample_uniform() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice()).run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<BigUint, _, _>::new(eqsat.egraph(), 5);
        let rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy
            .sample_eclass(&rng, 10, root_id, start_expr.len(), 4)
            .unwrap();

        assert_eq!(samples.len(), 6);
    }

    #[test]
    fn simple_sample_uniform_float() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice()).run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<f64, _, _>::new(eqsat.egraph(), 5);
        let rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy
            .sample_eclass(&rng, 10, root_id, start_expr.len(), 4)
            .unwrap();

        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn simple_greedy() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice()).run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedGreedy::<BigUint, _, _>::new(eqsat.egraph(), 5);
        let rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy
            .sample_eclass(&rng, 10, root_id, start_expr.len(), 4)
            .unwrap();

        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn halide_sample_uniform() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<BigUint, _, _>::new(eqsat.egraph(), 32);
        let rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_eclass(&rng, 10, root_id, 32, 4).unwrap();

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn halide_sample_greedy() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedGreedy::<BigUint, _, _>::new(eqsat.egraph(), 32);
        let rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy
            .sample_eclass(&rng, 10, root_id, start_expr.len(), 4)
            .unwrap();

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn halide_low_limit_greedy() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new((&start_expr).into(), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedGreedy::<BigUint, _, _>::new(eqsat.egraph(), 2);
        let rng = ChaCha12Rng::seed_from_u64(1024);

        assert_eq!(
            strategy.sample_eclass(&rng, 1000, root_id, start_expr.len(), 4),
            Err(SampleError::SizeLimit(13))
        );
    }

    #[test]
    fn combinations_2_10() {
        let combinations = sum_combinations(10, 2);
        assert_eq!(
            combinations,
            vec![
                vec![1, 9],
                vec![2, 8],
                vec![3, 7],
                vec![4, 6],
                vec![5, 5],
                vec![6, 4],
                vec![7, 3],
                vec![8, 2],
                vec![9, 1]
            ],
        );
    }

    #[test]
    fn combinations_3_5() {
        let combinations = sum_combinations(5, 3);
        assert_eq!(
            combinations,
            vec![
                vec![1, 1, 3],
                vec![1, 2, 2],
                vec![1, 3, 1],
                vec![2, 1, 2],
                vec![2, 2, 1],
                vec![3, 1, 1],
            ],
        );
    }

    #[test]
    fn combinations_0_0() {
        let combinations = sum_combinations(0, 0);
        assert_eq!(combinations, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn combinations_0_4() {
        let combinations = sum_combinations(0, 4);
        assert!(combinations.is_empty());
    }

    #[test]
    fn combinations_4_0() {
        let combinations = sum_combinations(4, 0);
        assert!(combinations.is_empty());
    }
}
