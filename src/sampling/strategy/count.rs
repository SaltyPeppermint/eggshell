use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{AddAssign, Mul};

use egg::{Analysis, AstSize, CostFunction, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;
use log::{debug, info, log_enabled, Level};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::WeightedError;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

use crate::analysis::commutative_semigroup::{CommutativeSemigroupAnalysis, Counter, ExprCount};
use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::sampling::choices::ChoiceList;
use crate::sampling::SampleError;

use super::Strategy;

/// Buggy budget consideration
#[derive(Debug)]
pub struct CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    egraph: &'a EGraph<L, N>,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    // flattened_size_counts: HashMap<Id, C>,
    ast_sizes: HashMap<Id, usize>,
    start_size: usize,
    limit: usize,
}

impl<'a, C, L, N> CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
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
    pub fn new_with_limit(egraph: &'a EGraph<L, N>, start_expr: &RecExpr<L>, limit: usize) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        // let limit = start_size + (start_size / 2);
        info!("Using limit {limit}");

        let mut size_counts = HashMap::new();
        // Make one big size count analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        ExprCount::new(limit).one_shot_analysis(egraph, &mut size_counts);
        info!("Size count oneshot analysis finsished!");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Strategy read to start sampling!");
        CountWeightedGreedy {
            egraph,
            size_counts,
            ast_sizes,
            start_size,
            limit,
        }
    }

    pub fn new(egraph: &'a EGraph<L, N>, start_expr: &RecExpr<L>) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        let limit = start_size + (start_size / 2);
        Self::new_with_limit(egraph, start_expr, limit)
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

impl<'a, C, L, N> Strategy<'a, L, N> for CountWeightedGreedy<'a, C, L, N>
where
    L: Language + Display + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        choices: &ChoiceList<L>,
    ) -> &'c L {
        if choices.len() <= self.start_size {
            self.pick_by_size_counts(rng, eclass)
        } else {
            pick_by_ast_size::<L, N>(&self.ast_sizes, eclass)
        }
    }

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.size_counts[&id].is_empty() {
            Err(SampleError::LimitError(self.limit))
        } else {
            Ok(())
        }
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }
}

#[derive(Debug)]
pub struct CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    egraph: &'a EGraph<L, N>,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    ast_sizes: HashMap<Id, usize>,
    limit: usize,
}

impl<'a, C, L, N> CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Display + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + Sum<C>
        + for<'x> Product<&'x C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    /// Creates a new [`CountWeightedUniformly<C, L, N>`].
    ///
    /// Terms are weighted according to the number of terms up to the size
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new_with_limit(egraph: &'a EGraph<L, N>, limit: usize) -> Self {
        // let limit = start_size + (start_size / 2);
        info!("Using limit {limit}");

        let mut size_counts = HashMap::new();
        // Make one big size count analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        ExprCount::new(limit).one_shot_analysis(egraph, &mut size_counts);
        info!("Size count oneshot analysis finsished!");

        // let mut max: C = 0u32.into();
        // for (_id, counts) in &size_counts {
        //     for (_size, count) in counts {
        //         if count > &max {
        //             max = count.clone();
        //         }
        //     }
        // }
        // println!("MAX ENCOUNTERED IS {max:?}");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Strategy read to start sampling!");
        CountWeightedUniformly {
            egraph,
            size_counts,
            ast_sizes,
            limit,
        }
    }

    pub fn new(egraph: &'a EGraph<L, N>, start_expr: &RecExpr<L>) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        let limit = start_size + (start_size / 2);
        Self::new_with_limit(egraph, limit)
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
                let combination_count = node
                    .children()
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
                    .unwrap_or_else(|| 0u32.into());
                combination_count
            })
            .sum::<C>()
    }
}

impl<'a, C, L, N> Strategy<'a, L, N> for CountWeightedUniformly<'a, C, L, N>
where
    L: Language + Display + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    C: Counter
        + Product<C>
        + Sum<C>
        + for<'x> Product<&'x C>
        + for<'x> Sum<&'x C>
        + for<'x> AddAssign<&'x C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        choices: &ChoiceList<L>,
    ) -> &'c L {
        debug!("Current EClass {:?}", eclass);
        debug!("Choices: {:?}", choices);
        // We need to know what is the minimum size required to fill the rest of the open positions
        let min_to_fill_other_open = choices
            .other_open_positions()
            .map(|id| self.ast_sizes[&id])
            .sum::<usize>();
        debug!("Required to fill rest: {min_to_fill_other_open}");
        // Budget for the children is one less because the node itself has size one at least
        // Also subtract the reserv budget needed for the other open positions
        let budget = self
            .limit
            .checked_sub(choices.n_chosen_positions() + min_to_fill_other_open)
            .unwrap();
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

                if log_enabled!(Level::Debug) {
                    let vec = budget_combinations.clone().collect::<Vec<Vec<usize>>>();
                    debug!("Budget combinations: {vec:?}");
                }

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

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.size_counts[&id].is_empty() {
            Err(SampleError::LimitError(self.limit))
        } else {
            Ok(())
        }
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }
}

/// Don't trust this, not really tested
#[derive(Debug)]
pub struct CountLutWeighted<'a, C, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
    C: Counter
        + Sum<C>
        + Product<C>
        + for<'x> AddAssign<&'x C>
        + for<'x> Mul<&'x C, Output = C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    egraph: &'a EGraph<L, N>,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    interesting_sizes: HashMap<usize, C>,
    ast_sizes: HashMap<Id, usize>,
    start_size: usize,
    limit: usize,
}

impl<'a, C, L, N> CountLutWeighted<'a, C, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
    C: Counter
        + Sum<C>
        + Product<C>
        + SampleUniform
        + PartialOrd
        + Default
        + for<'x> AddAssign<&'x C>
        + for<'x> Mul<&'x C, Output = C>,
{
    /// Creates a new [`CountLutWeighted<C, L, N>`].
    ///
    /// Terms are weighted according to their value in the hashtable.
    ///
    /// The analysis depth limit will be set to the largest term size in the table.
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new(
        egraph: &'a EGraph<L, N>,
        start_expr: &RecExpr<L>,
        interesting_sizes: HashMap<usize, C>,
    ) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        let mut size_counts = HashMap::new();
        let limit = *interesting_sizes
            .keys()
            .max()
            .expect("At least one term size of interest");

        // Make one big size counts analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        ExprCount::new(limit).one_shot_analysis(egraph, &mut size_counts);
        info!("Size count oneshot analysis finsished!");

        // Filter out data with uninteresting expression sizes
        info!("Filtering out uninteresting expression sizes...");
        for class_data in size_counts.values_mut() {
            class_data.retain(|k, _| interesting_sizes.contains_key(k));
        }
        info!("Filtering done!");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        info!("Strategy read to start sampling!");
        CountLutWeighted {
            egraph,
            size_counts,
            interesting_sizes,
            ast_sizes,
            start_size,
            limit,
        }
    }

    fn pick_by_lut<'c>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, <N as Analysis<L>>::Data>,
        budget: usize,
    ) -> &'c L {
        eclass
            .nodes
            .choose_weighted(rng, |node| -> C {
                node.children()
                    .iter()
                    .map(|child_id| &self.size_counts[child_id])
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| self.interesting_sizes[k].clone() * v)
                            .sum::<C>()
                    })
                    .product::<C>()
            })
            .or_else(|e| match e {
                // If all weights are zero, we are don't have
                // any data about the options to pick we are reasoning about.
                // If we have exhausted our budget we need to pick according
                // to AstSize now to "close out" the expr as fast as possible
                // to prevent it from growing even more.
                // Otherwise, go random.
                WeightedError::AllWeightsZero => {
                    if budget == 0 {
                        Ok(pick_by_ast_size::<L, N>(&self.ast_sizes, eclass))
                    } else {
                        Ok(eclass.nodes.choose(rng).expect("EClass cant be empty"))
                    }
                }
                _ => Err(e),
            })
            .expect("NoItem, InvalidWeight and TooMany variants should never trigger.")
    }
}

impl<'a, C, L, N> Strategy<'a, L, N> for CountLutWeighted<'a, C, L, N>
where
    L: Language + Display + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
    C: Counter
        + Sum<C>
        + Product<C>
        + for<'x> AddAssign<&'x C>
        + for<'x> Mul<&'x C, Output = C>
        + SampleUniform
        + PartialOrd
        + Default,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        choices: &ChoiceList<L>,
    ) -> &'c L {
        if choices.len() <= self.start_size {
            let budget = self.limit.saturating_sub(choices.len());
            self.pick_by_lut(rng, eclass, budget)
        } else {
            pick_by_ast_size::<L, N>(&self.ast_sizes, eclass)
        }
    }

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.size_counts[&id].is_empty() {
            Err(SampleError::LimitError(
                *self.interesting_sizes.keys().max().unwrap_or(&0),
            ))
        } else {
            Ok(())
        }
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
    use num::BigUint;
    use rand::SeedableRng;

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::sampling::SampleConf;
    use crate::trs::{Halide, Simple, TermRewriteSystem};

    use super::*;

    #[test]
    fn simple_sample_uniform() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<BigUint, _, _>::new_with_limit(eqsat.egraph(), 5);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy
            .sample_eclass(&mut rng, &sample_conf, root_id)
            .unwrap();

        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn simple_sample_uniform_float() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<f64, _, _>::new_with_limit(eqsat.egraph(), 5);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy
            .sample_eclass(&mut rng, &sample_conf, root_id)
            .unwrap();

        assert_eq!(samples.len(), 4);
    }

    #[test]
    fn simple_greedy() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy =
            CountWeightedGreedy::<BigUint, _, _>::new_with_limit(eqsat.egraph(), &start_expr, 5);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy
            .sample_eclass(&mut rng, &sample_conf, root_id)
            .unwrap();

        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn simple_sample_lut() {
        let term = "(* (+ a b) 1)";
        let start_expr = term.parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let strategy = CountLutWeighted::<BigUint, _, _>::new(
            eqsat.egraph(),
            &start_expr,
            HashMap::from([(1, 1u32.into()), (2, 2u32.into()), (17, 1u32.into())]),
        );
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy.sample_egraph(&mut rng, &sample_conf).unwrap();

        let n_samples = samples.iter().map(|(_, exprs)| exprs.len()).sum::<usize>();

        assert_eq!(n_samples, 8);
    }

    #[test]
    fn halide_sample_uniform() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy = CountWeightedUniformly::<BigUint, _, _>::new_with_limit(eqsat.egraph(), 32);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy
            .sample_eclass(&mut rng, &sample_conf, root_id)
            .unwrap();

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn halide_sample_greedy() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy =
            CountWeightedGreedy::<BigUint, _, _>::new_with_limit(eqsat.egraph(), &start_expr, 32);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
        let samples = strategy
            .sample_eclass(&mut rng, &sample_conf, root_id)
            .unwrap();

        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn halide_low_limit_greedy() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy =
            CountWeightedGreedy::<BigUint, _, _>::new_with_limit(eqsat.egraph(), &start_expr, 2);
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);

        assert_eq!(
            strategy.sample_eclass(&mut rng, &sample_conf, root_id),
            Err(SampleError::LimitError(2))
        );
    }

    #[test]
    fn halide_low_limit_lut() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let strategy = CountLutWeighted::<BigUint, _, _>::new(
            eqsat.egraph(),
            &start_expr,
            HashMap::from([(1, 1u32.into()), (2, 2u32.into()), (3, 1u32.into())]),
        );
        let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);

        assert_eq!(
            strategy.sample_eclass(&mut rng, &sample_conf, root_id),
            Err(SampleError::LimitError(3))
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
