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
    L: Language + Debug,
    N: Analysis<L> + Debug,
    C: Counter,
{
    egraph: &'a EGraph<L, N>,
    rng: &'b mut StdRng,
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
        + for<'x> Product<&'x C>
        + for<'x> Sum<&'x C>
        + SampleUniform
        + PartialOrd
        + Default
        + for<'x> AddAssign<&'x C>,
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
        info!("Starting oneshot analysis...");
        ExprCount::new(limit).one_shot_analysis(egraph, &mut size_counts);
        info!("Oneshot analysis finsished!");

        // Flatten for easier consumption
        info!("Flattening analysis data...");
        let flattened_size_counts = size_counts
            .iter()
            .map(|(k, v)| (*k, v.values().sum::<C>()))
            .collect();
        info!("Flattened analysis data!");

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Oneshot size count analysis finsished!");

        SizeCountWeighted {
            egraph,
            rng,
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
    ) -> &'c L {
        eclass
            .nodes
            .choose_weighted(&mut self.rng, |node| {
                node.children()
                    .iter()
                    .map(|child| &self.flattened_size_counts[child])
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
        + for<'x> Product<&'x C>
        + for<'x> Sum<&'x C>
        + SampleUniform
        + PartialOrd
        + Default
        + for<'x> AddAssign<&'x C>,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>, size: usize) -> &'c L {
        if size <= self.start_size {
            self.pick_by_size_counts(eclass)
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
    L: Language,
    N: Analysis<L>,
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
        let mut data = HashMap::new();
        let limit = *interesting_sizes
            .keys()
            .max()
            .expect("At least one term size of interest");

        // Make one big size counts analysis for all eclasses
        ExprCount::new(limit).one_shot_analysis(egraph, &mut data);
        // Filter out data with uninteresting term sizes
        for class_data in data.values_mut() {
            class_data.retain(|k, _| interesting_sizes.contains_key(k));
        }

        let mut ast_sizes = HashMap::new();
        // Make one big ast size analysis for all eclasses
        info!("Starting size count oneshot analysis...");
        AstSize.one_shot_analysis(egraph, &mut ast_sizes);
        info!("Oneshot size count analysis finsished!");

        SizeCountLutWeighted {
            egraph,
            rng,
            size_counts: data,
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
                    .map(|child| &self.size_counts[child])
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
        + SampleUniform
        + PartialOrd
        + Default
        + for<'x> AddAssign<&'x C>
        + for<'x> Mul<&'x C, Output = C>,
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
                .map(|child| ast_sizes[child])
                .min()
                .unwrap_or(0);
            (node, cost)
        })
        .min_by(|a, b| a.1.cmp(&b.1))
        .expect("EClasses can't have 0 members")
        .0
}

// let node_values = eclass
//     .nodes
//     .iter()
//     .map(|node| {
//         (
//             node,
//             node.children().iter().map(|child| &self.data[child]).fold(
//                 HashMap::new(),
//                 |mut lhs, rhs| {
//                     for (k, v) in rhs {
//                         lhs.entry(*k)
//                             .and_modify(|x| {
//                                 *x += v;
//                             })
//                             .or_insert(*v);
//                     }
//                     lhs
//                 },
//             ),
//         )
//     })
//     .collect::<Vec<_>>();

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
