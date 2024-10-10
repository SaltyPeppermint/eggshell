use std::fmt::Debug;

use egg::{Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language};
use hashbrown::HashMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::sampling::SampleError;

use super::Strategy;

#[derive(Debug)]
pub struct CostWeighted<'a, 'b, L, N, CF>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    egraph: &'a EGraph<L, N>,
    extractor: Extractor<'a, CF, L, N>,
    rng: &'b mut StdRng,
    loop_count: usize,
    raw_weights_memo: HashMap<Id, HashMap<&'a L, usize>>,
    loop_limit: usize,
}

impl<'a, 'b, L, N, CF> CostWeighted<'a, 'b, L, N, CF>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    /// Creates a new [`CostWeighted<'a, 'b, L, N, CF>`].
    ///
    pub fn new(
        egraph: &'a EGraph<L, N>,
        cost_fn: CF,
        rng: &'b mut StdRng,
        loop_limit: usize,
    ) -> Self {
        CostWeighted {
            egraph,
            extractor: Extractor::new(egraph, cost_fn),
            rng,
            loop_count: 0,
            loop_limit,
            raw_weights_memo: HashMap::new(),
        }
    }
}

impl<'a, 'b, L, N, CF> Strategy<'a, L, N> for CostWeighted<'a, 'b, L, N, CF>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>) -> &'c L {
        let pick = if self.loop_limit > self.loop_count {
            eclass
                .nodes
                .choose(&mut self.rng)
                .expect("Each class contains at least one enode.")
        } else {
            let raw_weights = self
                .raw_weights_memo
                .entry(eclass.id)
                .or_insert_with(|| calc_weights(eclass, &self.extractor));

            let urgency = (self.loop_count - self.loop_limit) as i32;

            let pick = if urgency < 32 {
                eclass
                    .nodes
                    .choose_weighted(&mut self.rng, |node| {
                        (raw_weights[node] as f64).powi(urgency)
                    })
                    .expect("Infallible weight calculation.")
            } else {
                eclass
                    .nodes
                    .iter()
                    .max_by_key(|node| raw_weights[node])
                    .unwrap()
            };
            pick
        };
        self.loop_count += 1;
        pick
    }

    fn extractable(&self, _id: Id) -> Result<(), SampleError> {
        Ok(())
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
    }

    fn rng_mut(&mut self) -> &mut StdRng {
        self.rng
    }

    fn start_new(&mut self) {
        self.loop_count = 0;
    }
}

fn calc_weights<'a, L, N, CF>(
    eclass: &'a EClass<L, N::Data>,
    extractor: &Extractor<CF, L, N>,
) -> HashMap<&'a L, usize>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Into<usize>,
{
    let costs = eclass.nodes.iter().map(|node| {
        node.children()
            .iter()
            .map(|c| extractor.find_best_cost(*c).into())
            .sum()
    });
    let max = costs.clone().max().unwrap_or(0);

    costs
        .zip(&eclass.nodes)
        .map(move |(cost, node)| (node, max - cost + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use egg::AstSize;
    use hashbrown::HashSet;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::eqsat::{Eqsat, EqsatConfBuilder, EqsatResult};
    use crate::sampling::SampleConfBuilder;
    use crate::trs::{Halide, Simple, Trs};

    use super::*;

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
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(12usize, n_samples);
    }

    #[test]
    fn stringified_sample_len() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = Simple::full_rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let mut n_samples = 0;
        let mut stringified = HashSet::new();
        for (_, exprs) in &samples {
            for expr in exprs {
                n_samples += 1;
                stringified.insert(format!("{expr}"));
            }
        }

        assert_eq!(n_samples, stringified.len());
    }

    #[test]
    fn multi_seed_sample() {
        let term = "(* (+ a b) 1)";
        let term2 = "(+ (+ x 0) (* y 1))";
        let seeds = vec![term.parse().unwrap(), term2.parse().unwrap()];
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = Simple::full_rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(seeds)
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(49usize, n_samples);
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
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(150usize, n_samples);
    }
}
