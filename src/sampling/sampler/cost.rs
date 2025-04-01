use std::fmt::{Debug, Display};

use egg::{Analysis, AstSize, CostFunction, EClass, EGraph, Extractor, Id, Language};
use hashbrown::HashMap;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;

use crate::sampling::choices::PartialRecExpr;

use super::Sampler;

/// Not well tested, do not use!
#[derive(Debug)]
pub struct CostWeighted<'a, L, N, CF>
where
    L: Language + Debug,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    egraph: &'a EGraph<L, N>,
    extractor: Extractor<'a, CF, L, N>,
}

impl<'a, L, N, CF> CostWeighted<'a, L, N, CF>
where
    L: Language,
    N: Analysis<L> + Debug,
    CF: CostFunction<L> + Debug,
    CF::Cost: Into<usize> + Debug,
{
    /// Creates a new [`CostWeighted<'a, 'b, L, N, CF>`].
    pub fn new(egraph: &'a EGraph<L, N>, cost_fn: CF) -> Self {
        CostWeighted {
            egraph,
            extractor: Extractor::new(egraph, cost_fn),
        }
    }
}

impl<'a, L, N, CF> Sampler<'a, L, N> for CostWeighted<'a, L, N, CF>
where
    L: Language + Display + Send + Sync,
    L::Discriminant: Sync,
    N: Analysis<L> + Debug + Sync,
    N::Data: Sync,
    CF: CostFunction<L> + Debug + Send + Sync,
    CF::Cost: Into<usize> + Send + Sync,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        size_limit: usize,
        partial_rec_expr: &PartialRecExpr<L>,
    ) -> &'c L {
        if partial_rec_expr.len() > size_limit {
            eclass
                .nodes
                .choose(rng)
                .expect("Each class contains at least one enode.")
        } else {
            let raw_weights = calc_weights(eclass, &self.extractor);

            eclass
                .nodes
                .iter()
                .max_by_key(|node| raw_weights[node])
                .unwrap()
        }
    }

    fn extractable(&self, id: Id, size_limit: usize) -> bool {
        Extractor::new(self.egraph, AstSize).find_best_cost(id) <= size_limit
    }

    fn analysis_depth(&self) -> Option<usize> {
        None
    }

    fn egraph(&self) -> &'a EGraph<L, N> {
        self.egraph
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
    use rand::SeedableRng;

    use super::*;
    use crate::eqsat::{Eqsat, EqsatConf, StartMaterial};
    use crate::trs::{Halide, Simple, TermRewriteSystem};

    #[test]
    fn simple_sample() {
        let start_expr = "(* (+ a b) 1)".parse().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&[&start_expr]), rules.as_slice()).run();

        let strategy = CostWeighted::new(eqsat.egraph(), AstSize);
        let mut rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_egraph(&mut rng, 1000, 4, 4).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(n_samples, 4);
    }

    #[test]
    fn stringified_sample_len() {
        let start_expr = "(* (+ a b) 1)".parse().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&[&start_expr]), rules.as_slice()).run();

        let strategy = CostWeighted::new(eqsat.egraph(), AstSize);
        let mut rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_egraph(&mut rng, 1000, 4, 4).unwrap();

        let mut n_samples = 0;
        let mut stringified = HashSet::new();
        for (_, exprs) in samples {
            for expr in exprs {
                n_samples += 1;
                stringified.insert(format!("{expr}"));
            }
        }

        assert_eq!(n_samples, stringified.len());
    }

    #[test]
    fn multi_seed_sample() {
        let start_expr_a = "(* (+ a b) 1)".parse().unwrap();
        let start_expr_b = "(+ (+ x 0) (* y 1))".parse().unwrap();

        let rules = Simple::full_rules();
        let eqsat = Eqsat::new(
            StartMaterial::RecExprs(&[&start_expr_a, &start_expr_b]),
            rules.as_slice(),
        )
        .run();

        let strategy = CostWeighted::new(eqsat.egraph(), AstSize);
        let mut rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_egraph(&mut rng, 1000, 4, 4).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(n_samples, 8);
    }

    #[test]
    fn halide_sample() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&[&start_expr]), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();

        let strategy = CostWeighted::new(eqsat.egraph(), AstSize);
        let mut rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_egraph(&mut rng, 1000, 64, 4).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(n_samples, 81);
    }

    #[test]
    fn halide_sample_float() {
        let start_expr = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )"
            .parse()
            .unwrap();
        let eqsat_conf = EqsatConf::builder().iter_limit(3).build();

        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&[&start_expr]), rules.as_slice())
            .with_conf(eqsat_conf)
            .run();

        let strategy = CostWeighted::new(eqsat.egraph(), AstSize);
        let mut rng = ChaCha12Rng::seed_from_u64(1024);
        let samples = strategy.sample_egraph(&mut rng, 1000, 64, 4).unwrap();

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(n_samples, 81);
    }
}
