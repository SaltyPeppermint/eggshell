use std::fmt::Display;

use egg::{Analysis, CostFunction, EGraph, Extractor, Id, Language, RecExpr};
use hashbrown::HashMap;
use rand::prelude::*;
use serde::Serialize;
use thiserror::Error;

use crate::{
    eqsat::{Eqsat, EqsatResult},
    trs::Trs,
};

#[derive(Error, Debug)]
pub enum SampleError {
    #[error("Batchsize impossible: {0}")]
    BatchSizeError(usize),
}

#[derive(Serialize, Debug, Clone)]
pub struct Sample<L: Language + Display> {
    expr: RecExpr<L>,
    eclass: Id,
    cost: usize,
}

#[allow(clippy::missing_errors_doc)]
pub fn sample<R: Trs>(
    seeds: &[RecExpr<R::Language>],
    batchsize: usize,
    samples_per_batch: usize,
    samples_per_eclass: usize,
) -> Result<Vec<Sample<R::Language>>, SampleError> {
    let rules = R::rules(&R::maximum_ruleset());
    let mut rng = StdRng::seed_from_u64(2024);

    if batchsize == 0 {
        return Err(SampleError::BatchSizeError(batchsize));
    }

    Ok(seeds
        .chunks(batchsize)
        .flat_map(|chunk| {
            let eqsat: EqsatResult<R> = Eqsat::new(chunk.to_vec()).with_explenation().run(&rules);
            let egraph = eqsat.egraph();
            extract_samples(egraph, samples_per_batch, samples_per_eclass, &mut rng)
        })
        .collect())
}

fn extract_samples<L: Language + Display, N: Analysis<L>>(
    egraph: &EGraph<L, N>,
    samples_per_batch: usize,
    samples_per_eclass: usize,
    rng: &mut StdRng,
) -> Vec<Sample<L>> {
    let mut samples = Vec::with_capacity(samples_per_batch * samples_per_eclass);
    let mut extract_history: HashMap<L, u32> = HashMap::new();
    for eclass in egraph.classes().choose_multiple(rng, samples_per_batch) {
        for _ in 0..samples_per_eclass {
            let cost_fn = FunkyCostFn::new(&extract_history);
            let extractor = Extractor::new(egraph, cost_fn);
            let (cost, expr) = extractor.find_best(eclass.id);
            drop(extractor);
            add_extracted(&mut extract_history, &expr, eclass.id);
            let sample = Sample {
                expr,
                eclass: eclass.id,
                cost,
            };
            samples.push(sample);
        }
    }

    samples
}

fn add_extracted<L: Language + std::hash::Hash>(
    prev_extracted: &mut HashMap<L, u32>,
    expr: &RecExpr<L>,
    root_id: Id,
) {
    fn rec_add<L: Language + std::hash::Hash>(
        prev_extracted: &mut HashMap<L, u32>,
        expr: &RecExpr<L>,
        id: Id,
    ) {
        let node = &expr[id];

        for child_id in node.children() {
            rec_add(prev_extracted, expr, *child_id);
        }

        // prev_extracted
        //     .entry(node)
        //     .and_modify(|c| *c += 1)
        //     .or_insert(1);

        if let Some(v) = prev_extracted.get_mut(node) {
            *v += 1;
        } else {
            prev_extracted.insert(node.clone(), 1);
        }
    }

    assert!(
        expr.is_dag(),
        "Something went horribly wrong, you cant extract something with cycles"
    );
    rec_add(prev_extracted, expr, root_id);
}

struct FunkyCostFn<'a, L: Language> {
    prev_extracted: &'a HashMap<L, u32>,
}

impl<'a, L: Language> FunkyCostFn<'a, L> {
    fn new(prev_extracted: &'a HashMap<L, u32>) -> Self {
        Self { prev_extracted }
    }
}

impl<'a, L: Language> CostFunction<L> for FunkyCostFn<'a, L> {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &L, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = self
            .prev_extracted
            .get(enode)
            .copied()
            .map_or(1, |n| 2usize.pow(n));
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}

// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
// struct NodeId {
//     class_id: Id,
//     node_idx: usize,
// }

// pub type Cost = NotNan<f64>;
// pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(std::f64::INFINITY) };
// pub const EPSILON_ALLOWANCE: f64 = 0.00001;

// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
// enum Status {
//     Doing,
//     Done,
// }

// #[derive(Default, Clone)]
// pub struct ExtractionResult {
//     pub choices: HashMap<Id, NodeId>,
// }

// impl ExtractionResult {
//     pub fn choose(&mut self, class_id: Id, node_id: NodeId) {
//         self.choices.insert(class_id, node_id);
//     }

//     pub fn node_sum_cost<L, CF>(
//         &self,
//         // egraph: &EGraph<L, N>,
//         node: &L,
//         costs: &HashMap<Id, Cost>,
//         cost_fn: &mut CF,
//     ) -> Cost
//     where
//         L: Language,
//         CF: CostFunction<L, Cost = Cost>,
//     {
//         let get_cost = |class_id| *costs.get(&class_id).unwrap_or(&INFINITY);
//         cost_fn.cost(node, get_cost)
//             + node
//                 .children()
//                 .iter()
//                 .map(|child_id| costs.get(child_id).unwrap_or(&INFINITY))
//                 .sum::<Cost>()
//     }
// }

// /// A faster bottom up extractor inspired by the faster-greedy-dag extractor.
// /// It should return an extraction result with the same cost as the bottom-up extractor.
// ///
// /// Bottom-up extraction works by iteratively computing the current best cost of each
// /// node in the e-graph based on the current best costs of its children.
// /// Extraction terminates when our estimates of the best cost for each node
// /// reach a fixed point.
// /// The baseline bottom-up implementation visits every node during each iteration
// /// of the fixed point.
// /// This algorithm instead only visits the nodes whose current cost estimate may change:
// /// it does this by tracking parent-child relationships and storing relevant nodes
// /// in a work list (UniqueQueue).
// pub struct FasterBottomUpExtractor;

// impl FasterBottomUpExtractor {
//     fn extract<L: Language, N: Analysis<L>, CF: CostFunction<L, Cost = Cost>>(
//         &self,
//         egraph: &EGraph<L, N>,
//         cost_fn: &mut CF,
//         _roots: &[Id],
//     ) -> ExtractionResult {
//         // let mut parents = HashMap::<Id, Vec<usize>>::with_capacity(egraph.classes().len());
//         // let n2c = |node_idx: &usize| egraph.nid_to_cid(node_idx);
//         let mut analysis_pending = UniqueQueue::default();

//         // Put in a child to look up its parent nodes identified by their class id and index in the parent class
//         let mut parents = egraph
//             .classes()
//             .map(|class| (class.id, Vec::new()))
//             .collect::<HashMap<Id, Vec<NodeId>>>();

//         for class in egraph.classes() {
//             for (node_idx, node) in class.nodes.iter().enumerate() {
//                 for child in node.children() {
//                     // compute parents of this enode
//                     if let Some(node_parents) = parents.get_mut(child) {
//                         node_parents.push(NodeId {
//                             class_id: class.id,
//                             node_idx,
//                         });
//                     }
//                 }

//                 // start the analysis from leaves
//                 if node.is_leaf() {
//                     analysis_pending.insert(NodeId {
//                         class_id: class.id,
//                         node_idx,
//                     });
//                 }
//             }
//         }

//         let mut result = ExtractionResult::default();
//         let mut costs = HashMap::<Id, Cost>::with_capacity(egraph.classes().len());

//         while let Some(node_id) = analysis_pending.pop() {
//             let NodeId { class_id, node_idx } = node_id;
//             let node = &egraph[class_id].nodes[node_idx];
//             let prev_cost = costs.get(&class_id).unwrap_or(&INFINITY);
//             let cost = result.node_sum_cost(node, &costs, cost_fn);
//             if cost < *prev_cost {
//                 result.choose(class_id.clone(), node_id);
//                 costs.insert(class_id.clone(), cost);
//                 analysis_pending.extend(parents[&class_id].iter().copied());
//             }
//         }

//         result
//     }
// }
