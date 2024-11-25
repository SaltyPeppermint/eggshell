use std::fmt::Debug;
use std::iter::{Product, Sum};
use std::ops::{AddAssign, Mul};

use egg::{Analysis, AstSize, CostFunction, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::HashMap;
use log::info;
use rand::distributions::uniform::SampleUniform;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::analysis::commutative_semigroup::{CommutativeSemigroupAnalysis, Counter, ExprCount};
use crate::analysis::semilattice::SemiLatticeAnalysis;
use crate::sampling::SampleError;

use super::Strategy;

#[derive(Debug)]
pub struct SizeCountWeighted<'a, 'b, C, L, N>
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
    rng: &'b mut StdRng,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    flattened_size_counts: HashMap<Id, C>,
    ast_sizes: HashMap<Id, usize>,
    start_size: usize,
    limit: usize,
}

impl<'a, 'b, C, L, N> SizeCountWeighted<'a, 'b, C, L, N>
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
    /// Creates a new [`TermNumberWeighted<L, N>`].
    ///
    /// Terms are weighted according to the number of terms up to the size
    /// cutoff limit
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new_with_limit(
        egraph: &'a EGraph<L, N>,
        rng: &'b mut StdRng,
        start_expr: &RecExpr<L>,
        limit: usize,
    ) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        // let limit = start_size + (start_size / 2);
        info!("Using limit {limit}");

        let mut size_counts = HashMap::new();
        // Make one big size count analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        ExprCount::new(limit).one_shot_analysis(egraph, &mut size_counts);
        info!("Size count oneshot analysis finsished!");

        // Flatten for easier consumption
        info!("Flattening analysis data...");
        let flattened_size_counts = size_counts
            .iter()
            .map(|(k, v)| (*k, v.values().sum::<C>()))
            .collect();
        info!("Flattened analysis data!");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting ast size oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Ast size oneshot analysis finsished!");

        SizeCountWeighted {
            egraph,
            rng,
            size_counts,
            flattened_size_counts,
            ast_sizes,
            start_size,
            limit,
        }
    }

    pub fn new(egraph: &'a EGraph<L, N>, rng: &'b mut StdRng, start_expr: &RecExpr<L>) -> Self {
        let start_size = AstSize.cost_rec(start_expr);
        let limit = start_size + (start_size / 2);
        Self::new_with_limit(egraph, rng, start_expr, limit)
    }

    fn pick_by_size_counts<'c>(
        &mut self,
        eclass: &'c EClass<L, <N as Analysis<L>>::Data>,
        budget: usize,
    ) -> &'c L {
        eclass
            .nodes
            .choose_weighted(&mut self.rng, |node| {
                node.children()
                    .iter()
                    .map(|child_id| {
                        self.size_counts[child_id]
                            .iter()
                            .filter(|(size, _)| size <= &&budget)
                            .map(|(_, cost)| cost)
                            .sum::<C>()
                    })
                    .product::<C>()
            })
            .unwrap()
    }

    #[must_use]
    pub fn flattened_size_counts(&self) -> &HashMap<Id, C> {
        &self.flattened_size_counts
    }
}

impl<'a, 'b, C, L, N> Strategy<'a, L, N> for SizeCountWeighted<'a, 'b, C, L, N>
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
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>, size: usize) -> &'c L {
        if size <= self.start_size {
            self.pick_by_size_counts(eclass, self.limit - size)
        } else {
            pick_by_ast_size::<L, N>(&self.ast_sizes, eclass)
        }
    }

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.flattened_size_counts[&id] == 0u32.into() {
            Err(SampleError::LimitError(self.limit))
        } else {
            Ok(())
        }
    }

    fn reset(&mut self) {}

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }

    fn rng_mut(&mut self) -> &mut StdRng {
        self.rng
    }
}

/// Don't trust this, not really tested
#[derive(Debug)]
pub struct SizeCountLutWeighted<'a, 'b, C, L, N>
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
    rng: &'b mut StdRng,
    size_counts: HashMap<Id, HashMap<usize, C>>,
    interesting_sizes: HashMap<usize, C>,
    ast_sizes: HashMap<Id, usize>,
    start_size: usize,
}

impl<'a, 'b, C, L, N> SizeCountLutWeighted<'a, 'b, C, L, N>
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
    /// Creates a new [`TermNumberWeighted<L, N>`].
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
        rng: &'b mut StdRng,
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

        SizeCountLutWeighted {
            egraph,
            rng,
            size_counts,
            interesting_sizes,
            ast_sizes,
            start_size,
        }
    }

    fn pick_by_lut<'c>(&mut self, eclass: &'c EClass<L, <N as Analysis<L>>::Data>) -> &'c L {
        eclass
            .nodes
            .choose_weighted(&mut self.rng, |node| -> C {
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
            .unwrap()
    }

    #[must_use]
    pub fn size_counts(&self) -> &HashMap<Id, HashMap<usize, C>> {
        &self.size_counts
    }
}

impl<'a, 'b, C, L, N> Strategy<'a, L, N> for SizeCountLutWeighted<'a, 'b, C, L, N>
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
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>, size: usize) -> &'c L {
        if size <= self.start_size {
            self.pick_by_lut(eclass)
        } else {
            pick_by_ast_size::<L, N>(&self.ast_sizes, eclass)
        }
    }

    fn reset(&mut self) {}

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

    fn rng_mut(&mut self) -> &mut StdRng {
        self.rng
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

#[cfg(test)]
mod tests {
    use num::BigUint;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::sampling::SampleConf;
    use crate::trs::{Halide, Simple, TermRewriteSystem, TrsEqsatResult};

    use super::*;

    #[test]
    fn simple_sample_lut() {
        let term = "(* (+ a b) 1)";
        let start_expr = term.parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> =
            Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
                .with_conf(eqsat_conf.clone())
                .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = SizeCountLutWeighted::<BigUint, _, _>::new(
            eqsat.egraph(),
            &mut rng,
            &start_expr,
            HashMap::from([(1, 1u32.into()), (2, 2u32.into()), (17, 1u32.into())]),
        );
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples = samples.iter().map(|(_, exprs)| exprs.len()).sum::<usize>();
        assert_eq!(8, n_samples);
    }

    #[test]
    fn simple_sample() {
        let start_expr = "(* (+ a b) 1)".parse::<RecExpr<_>>().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> =
            Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
                .with_conf(eqsat_conf.clone())
                .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = SizeCountWeighted::<BigUint, _, _>::new_with_limit(
            eqsat.egraph(),
            &mut rng,
            &start_expr,
            5,
        );
        let samples = strategy.sample_eclass(&sample_conf, root_id).unwrap();

        assert_eq!(7, samples.len());
    }

    #[test]
    fn halide_sample() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> =
            Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
                .with_conf(eqsat_conf.clone())
                .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = SizeCountWeighted::<BigUint, _, _>::new_with_limit(
            eqsat.egraph(),
            &mut rng,
            &start_expr,
            32,
        );

        let samples = strategy.sample_eclass(&sample_conf, root_id).unwrap();

        assert_eq!(10, samples.len());
    }

    #[test]
    fn halide_low_limit() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> =
            Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
                .with_conf(eqsat_conf.clone())
                .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = SizeCountWeighted::<BigUint, _, _>::new_with_limit(
            eqsat.egraph(),
            &mut rng,
            &start_expr,
            2,
        );
        assert_eq!(
            Err(SampleError::LimitError(2)),
            strategy.sample_eclass(&sample_conf, root_id)
        );
    }

    #[test]
    fn halide_lut_low_limit() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse::<RecExpr<_>>()
            .unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> =
            Eqsat::new(StartMaterial::RecExprs(vec![start_expr.clone()]))
                .with_conf(eqsat_conf.clone())
                .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = SizeCountLutWeighted::<BigUint, _, _>::new(
            eqsat.egraph(),
            &mut rng,
            &start_expr,
            HashMap::from([(1, 1u32.into()), (2, 2u32.into()), (3, 1u32.into())]),
        );
        assert_eq!(
            Err(SampleError::LimitError(3)),
            strategy.sample_eclass(&sample_conf, root_id)
        );
    }
}
