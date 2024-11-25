use std::fmt::{Debug, Display};

use egg::{Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language};
use hashbrown::HashMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::sampling::SampleError;

use super::Strategy;

/// Not well tested, do not use!
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
    raw_weights_memo: HashMap<Id, HashMap<&'a L, usize>>,
    limit: usize,
}

impl<'a, 'b, L, N, CF> CostWeighted<'a, 'b, L, N, CF>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    /// Creates a new [`CostWeighted<'a, 'b, L, N, CF>`].
    pub fn new(egraph: &'a EGraph<L, N>, cost_fn: CF, rng: &'b mut StdRng, limit: usize) -> Self {
        CostWeighted {
            egraph,
            extractor: Extractor::new(egraph, cost_fn),
            rng,
            limit,
            raw_weights_memo: HashMap::new(),
        }
    }
}

impl<'a, 'b, L, N, CF> Strategy<'a, L, N> for CostWeighted<'a, 'b, L, N, CF>
where
    L: Language + Display + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    fn pick<'c: 'a>(&mut self, eclass: &'c EClass<L, N::Data>, size: usize) -> &'c L {
        if size > self.limit {
            eclass
                .nodes
                .choose(&mut self.rng)
                .expect("Each class contains at least one enode.")
        } else {
            let raw_weights = self
                .raw_weights_memo
                .entry(eclass.id)
                .or_insert_with(|| calc_weights(eclass, &self.extractor));

            eclass
                .nodes
                .iter()
                .max_by_key(|node| raw_weights[node])
                .unwrap()
        }
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

    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::sampling::SampleConf;
    use crate::trs::{Halide, Simple, TermRewriteSystem, TrsEqsatResult};

    use super::*;

    #[test]
    fn simple_sample() {
        let start_expr = "(* (+ a b) 1)".parse().unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]))
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
        let start_expr = "(* (+ a b) 1)".parse().unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]))
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
        let start_exprs = vec![
            "(* (+ a b) 1)".parse().unwrap(),
            "(+ (+ x 0) (* y 1))".parse().unwrap(),
        ];
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::default();

        let rules = Simple::full_rules();
        let eqsat: TrsEqsatResult<Simple> = Eqsat::new(StartMaterial::RecExprs(start_exprs))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(46usize, n_samples);
    }

    #[test]
    fn halide_sample() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse()
            .unwrap();
        let sample_conf = SampleConf::default();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat: TrsEqsatResult<Halide> = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]))
            .with_conf(eqsat_conf.clone())
            .run(rules.as_slice());

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(107usize, n_samples);
    }
}
