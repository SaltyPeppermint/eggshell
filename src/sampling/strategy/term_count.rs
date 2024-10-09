use std::fmt::Debug;

use egg::{Analysis, EClass, EGraph, Id, Language};
use hashbrown::HashMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::analysis::commutative_semigroup::{CommutativeSemigroupAnalysis, TermsUpToSize};

use super::Strategy;

#[derive(Debug)]
pub struct TermCountWeighted<'a, 'b, L, N>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
{
    egraph: &'a EGraph<L, N>,
    rng: &'b mut StdRng,
    flattened_data: HashMap<Id, usize>,
}

impl<'a, 'b, L, N> TermCountWeighted<'a, 'b, L, N>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
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
    pub fn new(egraph: &'a EGraph<L, N>, rng: &'b mut StdRng, size_cutoff: usize) -> Self {
        let mut data = HashMap::new();

        // Make one big analysis for all eclasses
        TermsUpToSize::new(size_cutoff).one_shot_analysis(egraph, &mut data);
        // Filter out data with uninteresting term sizes

        let flattened_data = data
            .into_iter()
            .map(|(k, v)| (k, v.into_values().sum::<usize>()))
            .collect();

        TermCountWeighted {
            egraph,
            rng,
            flattened_data,
        }
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
                // 1 + to account for the term size of itself
                1 + node
                    .children()
                    .iter()
                    .map(|child| &self.flattened_data[child])
                    .sum::<usize>()
            })
            .unwrap()
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
    data: HashMap<Id, HashMap<usize, usize>>,
    interesting_sizes: HashMap<usize, usize>,
}

impl<'a, 'b, L, N> TermCountLutWeighted<'a, 'b, L, N>
where
    L: Language,
    N: Analysis<L>,
{
    /// Creates a new [`TermNumberWeighted<L, N>`].
    ///
    /// Terms are weighted according to their value in the hashtable
    ///
    ///
    /// # Panics
    ///
    /// Panics if given an empty Hashset of term sizes.
    pub fn new(
        egraph: &'a EGraph<L, N>,
        rng: &'b mut StdRng,
        interesting_sizes: HashMap<usize, usize>,
    ) -> Self {
        let mut data = HashMap::new();
        let max_term_size = interesting_sizes
            .keys()
            .max()
            .expect("At least one term size of interest");

        // Make one big analysis for all eclasses
        TermsUpToSize::new(*max_term_size).one_shot_analysis(egraph, &mut data);
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
                1 + node
                    .children()
                    .iter()
                    .map(|child| &self.data[child])
                    .map(|m| {
                        m.iter()
                            .map(|(k, v)| self.interesting_sizes[k] * v)
                            .sum::<usize>()
                    })
                    .sum::<usize>()
            })
            .unwrap()
    }

    fn start_new(&mut self) {}

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

    use crate::eqsat::{Eqsat, EqsatConfBuilder, EqsatResult};
    use crate::sampling::SampleConfBuilder;
    use crate::trs::{Halide, Simple, Trs};

    use super::*;

    #[test]
    fn simple_sample_lut() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = Simple::full_rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountLutWeighted::new(
            eqsat.egraph(),
            &mut rng,
            HashMap::from([(15, 1), (16, 2), (17, 1)]),
        );
        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(19usize, n_samples);
    }

    #[test]
    fn simple_sample() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = Simple::full_rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 5);
        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(16usize, n_samples);
    }

    #[test]
    fn halide_sample() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat: EqsatResult<Halide> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 8);

        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(152usize, n_samples);
    }

    #[test]
    fn halide_term_min_size() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat: EqsatResult<Halide> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy = TermCountWeighted::new(eqsat.egraph(), &mut rng, 8);

        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(152usize, n_samples);
    }
}
