mod choice;
mod utils;

use std::fmt::{Debug, Display};
use std::iter::IntoIterator;
use std::iter::Sum;
use std::ops::AddAssign;

use egg::{Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language, RecExpr};
use rand::distributions::uniform::{SampleBorrow, SampleUniform};
use rand::prelude::*;
use serde::Serialize;
use thiserror::Error;

use crate::eqsat::utils::EqsatConf;
use crate::eqsat::{Eqsat, EqsatResult};
use crate::trs::Trs;
use crate::utils::AstSize2;
use crate::{HashMap, HashSet};
use choice::Choice;

pub use utils::{SampleConf, SampleConfBuilder};

#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Batchsize impossible: {0}")]
    BatchSizeError(usize),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Sample<L: Language + Display> {
    seed_exprs: RecExpr<L>,
    samples: HashMap<Id, HashSet<RecExpr<L>>>,
    sample_conf: SampleConf,
    eqsat_conf: EqsatConf,
}

#[allow(clippy::missing_errors_doc)]
pub fn sample_multiple<R: Trs>(
    seed_expr: Vec<RecExpr<R::Language>>,
    sample_conf: &SampleConf,
    eqsat_conf: &EqsatConf,
) -> Result<Vec<Sample<R::Language>>, SampleError> {
    seed_expr
        .into_iter()
        .map(|seed| sample::<R>(seed, sample_conf, eqsat_conf))
        .collect()
}

#[allow(clippy::missing_errors_doc)]
pub fn sample<R: Trs>(
    seed_expr: RecExpr<R::Language>,
    sample_conf: &SampleConf,
    eqsat_conf: &EqsatConf,
) -> Result<Sample<R::Language>, SampleError> {
    let rules = R::rules(&R::maximum_ruleset());
    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    if eqsat_conf.explenation {
        println!("Running without explenation");
    }

    let eqsat: EqsatResult<R> = Eqsat::new(vec![seed_expr.clone()])
        .with_conf(eqsat_conf.clone())
        .run(&rules);
    let egraph = eqsat.egraph();
    let samples = sample_egrpah(egraph, sample_conf, &mut rng);

    Ok(Sample {
        seed_exprs: seed_expr,
        samples,
        sample_conf: sample_conf.clone(),
        eqsat_conf: eqsat_conf.clone(),
    })
}

fn sample_egrpah<L: Language, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    conf: &SampleConf,
    rng: &mut StdRng,
) -> HashMap<Id, HashSet<RecExpr<L>>> {
    let extractor = Extractor::new(egraph, AstSize2);

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
            // HOPEFULLY NOW NO LONGER NEEDED with better `From` Implementation
            // See sample_terms
            // This deduplicates the expressions.
            // Dedup only removes consecutive duplicates so we need to sort first
            // exprs.sort();
            // exprs.dedup();
            (eclass.id, exprs)
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
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
    let mut choices = Choice::Open(root_eclass.id);
    let mut visited = HashSet::from([root_eclass.id]);
    let mut loop_count = 0;

    while let Some(next_open) = choices.next_open() {
        let eclass_id = egraph.find(next_open.eclass_id());
        let eclass = &egraph[eclass_id];
        let pick = if loop_limit > loop_count {
            eclass.nodes.choose(rng).unwrap()
        } else {
            let raw_weights = raw_weights_memo
                .entry(eclass.id)
                .or_insert_with(|| calc_weights(eclass, extractor));

            let urgency = f64::sqrt((loop_count - loop_limit) as f64);

            let pick = eclass
                .nodes
                .choose_weighted(rng, |node| (raw_weights[node] as f64).powf(urgency))
                .expect("Infallible weight calculation.");
            pick
        };

        visited.insert(eclass_id);

        if pick
            .children()
            .iter()
            .any(|child_id| visited.contains(child_id))
        {
            loop_count += 1;
        }
        *next_open = Choice::Picked {
            eclass_id,
            pick,
            children: pick
                .children()
                .iter()
                .map(|child_id| Choice::Open(*child_id))
                .collect(),
        }
    }
    // HOPEFULLY NOW NO LONGER NEEDED with better `From` Implementation
    // See sample_egraph
    // this is a dirty hack to get dedup easily
    // Essentiall, we are reassigning the Ids canonically.
    // Should be doable with a simpler expression and then we wouldnt need to use the sort+dedup trick later
    // And could rely on a hashset
    //
    // let s = format!("{}", RecExpr::from(choices));
    // RecExpr::from_str(&s).unwrap()

    RecExpr::from(choices)
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

    use crate::eqsat::utils::EqsatConfBuilder;
    use crate::trs::{Halide, Simple};

    use super::*;
    use utils::SampleConfBuilder;

    #[test]
    fn simple_sample() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sampel_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let sample = sample::<Simple>(seed, &sampel_conf, &eqsat_conf).unwrap();

        let n_samples = sample.samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(11usize, n_samples);
    }

    #[test]
    fn stringified_sample_len() {
        let term = "(* (+ a b) 1)";
        let seed = term.parse().unwrap();
        let sampel_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let sample = sample::<Simple>(seed, &sampel_conf, &eqsat_conf).unwrap();

        let mut n_samples = 0;
        let mut stringified = HashSet::new();
        for (_, exprs) in &sample.samples {
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
        let sampel_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new().build();

        let samples = sample_multiple::<Simple>(seeds, &sampel_conf, &eqsat_conf).unwrap();

        let n_samples = samples
            .iter()
            .flat_map(|sample| sample.samples.iter().map(|(_, exprs)| exprs.len()))
            .sum();

        assert_eq!(49usize, n_samples);
    }

    #[test]
    fn halide_sample() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seed = term.parse().unwrap();
        let sampel_conf = SampleConfBuilder::new().build();
        let eqsat_conf = EqsatConfBuilder::new()
            .time_limit(Duration::from_secs_f64(0.2))
            .build();

        let sample = sample::<Halide>(seed, &sampel_conf, &eqsat_conf).unwrap();

        // let mut n_samples = 0;
        // for (eclass_id, exprs) in &sample.samples {
        //     for expr in exprs {
        //         n_samples += 1;
        //         // println!("{}: {eclass_id}: {expr}", &sample.seed_exprs);
        //     }
        // }
        let n_samples = sample.samples.iter().map(|(_, exprs)| exprs.len()).sum();
        assert_eq!(256usize, n_samples);
    }
}
