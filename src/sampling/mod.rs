mod choices;
pub mod strategy;
mod utils;

use std::fmt::{Debug, Display};

use egg::{Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use serde::Serialize;
use thiserror::Error;

use crate::eqsat::EqsatConf;

pub use utils::{SampleConf, SampleConfBuilder};

#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Batchsize impossible: {0}")]
    BatchSizeError(usize),
    #[error("Can't convert a non-finished list of choices")]
    ChoiceError,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Sample<L: Language + Display> {
    seed_exprs: RecExpr<L>,
    samples: HashMap<Id, HashSet<RecExpr<L>>>,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use egg::AstSize;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::eqsat::{Eqsat, EqsatConfBuilder, EqsatResult};
    use crate::trs::{Halide, Simple, Trs};
    use strategy::Strategy;

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
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            strategy::Uniform::new(&mut rng, eqsat.egraph(), AstSize, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf);

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
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            strategy::Uniform::new(&mut rng, eqsat.egraph(), AstSize, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf);

        let mut n_samples = 0;
        let mut stringified = HashSet::new();
        for (_, exprs) in &samples {
            for expr in exprs {
                // println!("{}: {eclass_id}: {expr}", &sample.seed_exprs);
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
        let eqsat: EqsatResult<Simple> =
            Eqsat::new(seeds).with_conf(eqsat_conf.clone()).run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            strategy::Uniform::new(&mut rng, eqsat.egraph(), AstSize, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(49usize, n_samples);
    }

    #[test]
    fn halide_sample() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new()
            .time_limit(Duration::from_secs_f64(0.2))
            .build();

        let rules = Halide::full_rules();
        let eqsat: EqsatResult<Halide> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let mut strategy =
            strategy::Uniform::new(&mut rng, eqsat.egraph(), AstSize, sample_conf.loop_limit);
        let samples = strategy.sample(&sample_conf);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(256usize, n_samples);
    }
}
