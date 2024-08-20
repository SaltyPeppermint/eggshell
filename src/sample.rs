use std::fmt::{Debug, Display};
use std::iter::IntoIterator;
use std::iter::Sum;
use std::ops::AddAssign;

use egg::{Analysis, CostFunction, EClass, EGraph, Extractor, Id, Language, RecExpr};
use rand::distributions::uniform::{SampleBorrow, SampleUniform};
use rand::prelude::*;
use serde::Serialize;
use thiserror::Error;

use crate::eqsat::{Eqsat, EqsatResult};
use crate::trs::Trs;
use crate::utils::AstSize2;
use crate::{HashMap, HashSet};

#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Batchsize impossible: {0}")]
    BatchSizeError(usize),
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sample<L: Language + Display> {
    expr: RecExpr<L>,
    eclass: Id,
}

#[allow(clippy::missing_errors_doc)]
pub fn sample<R: Trs>(
    seed_expr: &[RecExpr<R::Language>],
    seeds_per_egraph: usize,
    samples_per_egraph: usize,
    samples_per_eclass: usize,
    loop_limit: usize,
    rng_seed: u64,
) -> Result<Vec<Sample<R::Language>>, SampleError> {
    let rules = R::rules(&R::maximum_ruleset());
    let mut rng = StdRng::seed_from_u64(rng_seed);

    if seeds_per_egraph == 0 {
        return Err(SampleError::BatchSizeError(seeds_per_egraph));
    }

    Ok(seed_expr
        .chunks(seeds_per_egraph)
        .flat_map(|chunk| {
            let eqsat: EqsatResult<R> = Eqsat::new(chunk.to_vec()).with_explenation().run(&rules);
            let egraph = eqsat.egraph();
            sample_egrpah(
                egraph,
                samples_per_egraph,
                samples_per_eclass,
                loop_limit,
                &mut rng,
            )
        })
        .collect())
}

fn sample_egrpah<L: Language + Display, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    samples_per_egraph: usize,
    samples_per_eclass: usize,
    loop_limit: usize,
    rng: &mut StdRng,
) -> HashSet<Sample<L>> {
    //let mut samples = Vec::with_capacity(samples_per_batch * samples_per_eclass);
    let extractor = Extractor::new(egraph, AstSize2);

    let mut samples = HashSet::new();
    let mut raw_weights_memo = HashMap::new();
    for eclass in egraph.classes().choose_multiple(rng, samples_per_egraph) {
        for _ in 0..samples_per_eclass {
            let expr = sample_term(
                egraph,
                eclass,
                &extractor,
                loop_limit,
                rng,
                &mut raw_weights_memo,
            );
            samples.insert(Sample {
                expr,
                eclass: eclass.id,
            });
        }
    }
    samples
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq, Hash)]
enum Choice<'a, L: Language> {
    Open(Id),
    Picked {
        eclass_id: Id,
        pick: &'a L,
        children: Vec<Choice<'a, L>>,
    },
}

impl<'a, L: Language> Choice<'a, L> {
    fn eclass_id(&self) -> Id {
        match self {
            Choice::Picked { eclass_id, .. } | Choice::Open(eclass_id) => *eclass_id,
        }
    }

    fn children(&self) -> &'a [Choice<L>] {
        match self {
            Choice::Open(_) => &[],
            Choice::Picked { children, .. } => children,
        }
    }

    // fn pick(&self) -> Option<&'a L> {
    //     match self {
    //         Choice::Open(_) => None,
    //         Choice::Picked { pick, .. } => Some(pick),
    //     }
    // }

    fn collect_children(self, all: &mut Vec<(Id, L)>) {
        match self {
            Choice::Open(_) => {
                panic!("Calling collect on an unfinished tree makes no sense")
            }
            Choice::Picked {
                eclass_id,
                pick,
                children,
            } => {
                for child in children {
                    child.collect_children(all);
                }
                all.push((eclass_id, pick.clone()));
            }
        }
    }

    fn all_choices(&'a self, all: &mut Vec<&'a Choice<'a, L>>) {
        all.push(self);
        for child in self.children() {
            child.all_choices(all);
        }
    }

    fn next_open(&mut self) -> Option<&mut Self> {
        match self {
            Choice::Open(_) => Some(self),
            Choice::Picked { children, .. } => children.iter_mut().find_map(|c| c.next_open()),
        }
    }
}

impl<'a, L: Language> IntoIterator for &'a Choice<'a, L> {
    type Item = &'a Choice<'a, L>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut all_children = Vec::new();
        self.all_choices(&mut all_children);
        all_children.into_iter()
    }
}

impl<'a, L: Language> From<Choice<'a, L>> for RecExpr<L> {
    fn from(choices: Choice<'a, L>) -> Self {
        let mut picks = Vec::new();
        let mut translation_table = HashMap::new();
        let mut id_counter = 0;

        choices.collect_children(&mut picks);

        let mut expr = RecExpr::default();

        for (id, mut node) in picks {
            {
                translation_table.entry(id).or_insert_with(|| {
                    let new_id = Id::from(id_counter);
                    id_counter += 1;
                    new_id
                });
                for child_id in node.children_mut() {
                    *child_id = translation_table[child_id];
                }
                expr.add(node);
            }
        }
        expr
    }
}

#[allow(clippy::cast_precision_loss)]
fn sample_term<'b, L, N, CF, X>(
    egraph: &'b EGraph<L, N>,
    root_eclass: &EClass<L, N::Data>,
    extractor: &'b Extractor<CF, L, N>,
    loop_limit: usize,
    rng: &mut StdRng,
    raw_weights_memo: &mut HashMap<Id, HashMap<&'b L, usize>>,
) -> RecExpr<L>
where
    L: Language,
    N: Analysis<L>,
    CF: CostFunction<L>,
    CF::Cost: Sum + SampleBorrow<X> + Into<usize>,
    X: SampleUniform + for<'a> AddAssign<&'a X> + PartialOrd<X> + Clone + Default,
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
    use crate::trs::{Halide, Simple};

    use super::*;

    #[test]
    fn simple_sample() {
        let term = "(* (+ a b) 1)";
        let seeds = [term.parse().unwrap()];

        let samples = sample::<Simple>(&seeds, 2, 2, 2, 4, 2024).unwrap();

        for sample in &samples {
            println!("{}: {}", sample.eclass, sample.expr);
        }

        assert_eq!(2, samples.len());
    }

    #[test]
    fn multi_seed_sample() {
        let term = "(* (+ a b) 1)";
        let term2 = "(+ (+ x 0) (* y 1))";
        let seeds = [term.parse().unwrap(), term2.parse().unwrap()];

        let samples = sample::<Simple>(&seeds, 2, 2, 2, 4, 2024).unwrap();

        for sample in &samples {
            println!("{}: {}", sample.eclass, sample.expr);
        }

        assert_eq!(3, samples.len());
    }

    #[test]
    fn halide_sample() {
        let term = "( >= ( + ( + v0 v1 ) v2 ) ( + ( + ( + v0 v1 ) v2 ) 1 ) )";
        let seeds = [term.parse().unwrap()];

        let samples = sample::<Halide>(&seeds, 2, 4, 32, 8, 2024).unwrap();

        for sample in &samples {
            println!("{}: {}", sample.eclass, sample.expr);
        }

        assert_eq!(7, samples.len());
    }
}
