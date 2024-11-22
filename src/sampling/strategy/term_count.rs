use std::fmt::Debug;

use egg::{Analysis, EClass, EGraph, Id, Language};
use hashbrown::HashMap;
use log::info;
use num::BigUint;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::analysis::commutative_semigroup::{CommutativeSemigroupAnalysis, TermCount};
use crate::sampling::SampleError;

use super::Strategy;

#[derive(Debug)]
pub struct TermCountWeighted<'a, 'b, L, N>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
{
    egraph: &'a EGraph<L, N>,
    rng: &'b mut StdRng,
    flattened_data: HashMap<Id, BigUint>,
    limit: usize,
}

impl<'a, 'b, L, N> TermCountWeighted<'a, 'b, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
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
    pub fn new(egraph: &'a EGraph<L, N>, rng: &'b mut StdRng, limit: usize) -> Self {
        let mut data = HashMap::new();

        // Make one big analysis for all eclasses
        info!("Starting oneshot analysis...");
        TermCount::new(limit).one_shot_analysis(egraph, &mut data);
        info!("Oneshot analysis finsished!");

        // Flatten for easier consumption
        info!("Flattening analysis data...");
        let flattened_data = data
            .into_iter()
            .map(|(k, v)| (k, v.values().sum::<BigUint>()))
            .collect();
        info!("Flattened analysis data!");

        TermCountWeighted {
            egraph,
            rng,
            flattened_data,
            limit,
        }
    }

    #[must_use]
    pub fn flattened_data(&self) -> &HashMap<Id, BigUint> {
        &self.flattened_data
    }
}

impl<'a, 'b, L, N> Strategy<'a, L, N> for TermCountWeighted<'a, 'b, L, N>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>) -> &'c L {
        eclass
            .nodes
            .choose_weighted(&mut self.rng, |node| {
                node.children()
                    .iter()
                    .map(|child| &self.flattened_data[child])
                    .product::<BigUint>()
            })
            .unwrap()
    }

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.flattened_data[&id] == 0usize.into() {
            Err(SampleError::LimitError(self.limit))
        } else {
            Ok(())
        }
    }

    fn start_new(&mut self) {}

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }

    fn rng_mut(&mut self) -> &mut StdRng {
        self.rng
    }
}

#[derive(Debug)]
pub struct TermCountLutWeighted<'a, 'b, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    egraph: &'a EGraph<L, N>,
    rng: &'b mut StdRng,
    data: HashMap<Id, HashMap<usize, BigUint>>,
    interesting_sizes: HashMap<usize, BigUint>,
}

impl<'a, 'b, L, N> TermCountLutWeighted<'a, 'b, L, N>
where
    L: Language + Debug + Sync + Send,
    L::Discriminant: Debug + Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Debug + Sync,
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
        interesting_sizes: HashMap<usize, BigUint>,
    ) -> Self {
        let mut data = HashMap::new();
        let max_term_size = interesting_sizes
            .keys()
            .max()
            .expect("At least one term size of interest");

        // Make one big analysis for all eclasses
        TermCount::new(*max_term_size).one_shot_analysis(egraph, &mut data);
        // Filter out data with uninteresting term sizes
        for class_data in data.values_mut() {
            class_data.retain(|k, _| interesting_sizes.contains_key(k));
        }

        TermCountLutWeighted {
            egraph,
            rng,
            data,
            interesting_sizes,
        }
    }

    #[must_use]
    pub fn data(&self) -> &HashMap<Id, HashMap<usize, BigUint>> {
        &self.data
    }
}

impl<'a, 'b, L, N> Strategy<'a, L, N> for TermCountLutWeighted<'a, 'b, L, N>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>) -> &'c L {
        eclass
            .nodes
            .choose_weighted(&mut self.rng, |node| {
                // 1 + to account for the term size of itself
                BigUint::from(1usize)
                    + node
                        .children()
                        .iter()
                        .map(|child| &self.data[child])
                        .map(|m| {
                            m.iter()
                                .map(|(k, v)| self.interesting_sizes[k].clone() * v)
                                .sum::<BigUint>()
                        })
                        .sum::<BigUint>()
            })
            .unwrap()
    }

    fn start_new(&mut self) {}

    fn extractable(&self, id: Id) -> Result<(), SampleError> {
        if self.data[&id].is_empty() {
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::sampling::SampleConf;
    use crate::trs::{Halide, Simple, TermRewriteSystem, TrsEqsatResult};

    use super::*;

    #[test]
    fn simple_sample_lut() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> = Eqsat::new(StartMaterial::Seeds(vec![seed]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountLutWeighted::new(
            eqsat.egraph(),
            &mut rng,
            HashMap::from([(1, 1usize.into()), (2, 2usize.into()), (17, 1usize.into())]),
        );
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(13usize, n_samples);
    }

    #[test]
    fn simple_sample() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> = Eqsat::new(StartMaterial::Seeds(vec![seed]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 5);
        let samples = strategy.sample_eclass(&sample_conf, root_id).unwrap();

        assert_eq!(10usize, samples.len());
    }

    #[test]
    fn halide_sample() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConf::builder().samples_per_eclass(10).build();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> = Eqsat::new(StartMaterial::Seeds(vec![seed]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 32);

        let samples = strategy.sample_eclass(&sample_conf, root_id).unwrap();

        assert_eq!(10usize, samples.len());
    }

    #[test]
    fn halide_low_limit() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> = Eqsat::new(StartMaterial::Seeds(vec![seed]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 2);
        assert_eq!(
            Err(SampleError::LimitError(2)),
            strategy.sample_eclass(&sample_conf, root_id)
        );
    }

    #[test]
    fn halide_lut_low_limit() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(2).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> = Eqsat::new(StartMaterial::Seeds(vec![seed]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());
        let root_id = eqsat.roots()[0];

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountLutWeighted::new(
            eqsat.egraph(),
            &mut rng,
            HashMap::from([(1, 1usize.into()), (2, 2usize.into()), (3, 1usize.into())]),
        );
        assert_eq!(
            Err(SampleError::LimitError(3)),
            strategy.sample_eclass(&sample_conf, root_id)
        );
    }
}
