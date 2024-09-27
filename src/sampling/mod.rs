mod choices;
mod utils;

use std::fmt::{Debug, Display};
use std::iter::IntoIterator;
use std::iter::Sum;
use std::ops::AddAssign;

use egg::{Analysis, AstSize, CostFunction, EClass, EGraph, Extractor, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use rand::distributions::uniform::{SampleBorrow, SampleUniform};
use rand::prelude::*;
use serde::Serialize;
use thiserror::Error;

use crate::eqsat::EqsatConf;
use choices::ChoiceList;

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

pub fn sample<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    conf: &SampleConf,
    rng: &mut StdRng,
) -> HashMap<Id, HashSet<RecExpr<L>>> {
    let extractor = Extractor::new(egraph, AstSize);

    let mut raw_weights_memo = HashMap::new();
    egraph
        .classes()
        .choose_multiple(rng, conf.samples_per_egraph)
        .into_iter()
        .map(|eclass| {
            let exprs = (0..conf.samples_per_eclass)
                .map(|_| {
                    sample_term(
                        egraph,
                        eclass,
                        &extractor,
                        conf.loop_limit,
                        rng,
                        &mut raw_weights_memo,
                    )
                })
                .collect();
            (eclass.id, exprs)
        })
        .collect()
}

pub fn sample_root<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    conf: &SampleConf,
    root: Id,
    rng: &mut StdRng,
) -> HashSet<RecExpr<L>> {
    let extractor = Extractor::new(egraph, AstSize);

    let mut raw_weights_memo = HashMap::new();

    (0..conf.samples_per_eclass)
        .map(|_| {
            sample_term(
                egraph,
                &egraph[root],
                &extractor,
                conf.loop_limit,
                rng,
                &mut raw_weights_memo,
            )
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn sample_term<'a, L, N, CF, X>(
    egraph: &'a EGraph<L, N>,
    root_eclass: &EClass<L, N::Data>,
    extractor: &'a Extractor<CF, L, N>,
    loop_limit: usize,
    rng: &mut StdRng,
    raw_weights_memo: &mut HashMap<Id, HashMap<&'a L, usize>>,
) -> RecExpr<L>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Sum + SampleBorrow<X> + Into<usize>,
    X: SampleUniform + for<'x> AddAssign<&'x X> + PartialOrd<X> + Clone + Default,
{
    let mut choices: ChoiceList<L> = ChoiceList::from(root_eclass.id);
    // let mut visited = HashSet::from([root_eclass.id]);
    let mut loop_count = 0;

    while let Some(next_open_id) = choices.next_open() {
        let eclass_id = egraph.find(next_open_id);
        let eclass = &egraph[eclass_id];
        let pick = if loop_limit > loop_count {
            eclass
                .nodes
                .choose(rng)
                .expect("Each class contains at least one enode.")
        } else {
            let raw_weights = raw_weights_memo
                .entry(eclass.id)
                .or_insert_with(|| calc_weights(eclass, extractor));

            let urgency = (loop_count - loop_limit) as i32;
            // println!("Urgency: {urgency}");
            // println!("{raw_weights:?}");
            let pick = if urgency < 32 {
                eclass
                    .nodes
                    .choose_weighted(rng, |node| (raw_weights[node] as f64).powi(urgency))
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

        // visited.insert(eclass_id);

        // if pick
        //     .children()
        //     .iter()
        //     .any(|child_id| visited.contains(child_id))
        // {
        //     loop_count += 1;
        // }
        loop_count += 1;
        choices.fill_next(pick);
    }
    choices.try_into().expect("No open choices should be left")
}

fn calc_weights<'a, L, N, CF>(
    eclass: &'a EClass<L, N::Data>,
    extractor: &'a Extractor<CF, L, N>,
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
    use std::time::Duration;

    use crate::eqsat::EqsatConfBuilder;
    use crate::eqsat::{Eqsat, EqsatResult};
    use crate::trs::{Halide, Ruleset, Simple, Trs};

    use super::*;

    #[test]
    fn simple_sample() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = <Simple as Trs>::Rules::Full.rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let samples = sample(eqsat.egraph(), &sample_conf, &mut rng);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(12usize, n_samples);
    }

    #[test]
    fn stringified_sample_len() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sample_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let rules = <Simple as Trs>::Rules::Full.rules();
        let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let samples = sample(eqsat.egraph(), &sample_conf, &mut rng);

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

        let rules = <Simple as Trs>::Rules::Full.rules();
        let eqsat: EqsatResult<Simple> =
            Eqsat::new(seeds).with_conf(eqsat_conf.clone()).run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let samples = sample(eqsat.egraph(), &sample_conf, &mut rng);

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

        let rules = <Halide as Trs>::Rules::Full.rules();
        let eqsat: EqsatResult<Halide> = Eqsat::new(vec![seed])
            .with_conf(eqsat_conf.clone())
            .run(&rules);

        let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
        let samples = sample(eqsat.egraph(), &sample_conf, &mut rng);

        let n_samples: usize = samples.iter().map(|(_, exprs)| exprs.len()).sum();

        assert_eq!(256usize, n_samples);
    }
}
