use egg::{Id, Language, RecExpr};
use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::Serialize;

use crate::meta_lang::ProbabilisticLang;

#[gen_stub_pyclass]
#[pyclass(frozen, module = "eggshell")]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct FirstErrorDistance {
    hits: HashSet<Id>,
    misses: HashSet<Id>,
    #[pyo3(get)]
    avg_hit_confidence: Option<f64>,
    #[pyo3(get)]
    avg_miss_confidence: Option<f64>,
}

impl Default for FirstErrorDistance {
    fn default() -> Self {
        Self {
            hits: HashSet::new(),
            misses: HashSet::new(),
            avg_hit_confidence: None,
            avg_miss_confidence: None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FirstErrorDistance {
    #[getter(misses)]
    pub fn __misses__(&self) -> Vec<usize> {
        self.misses.iter().map(|id| (*id).into()).collect()
    }

    #[getter(hits)]
    pub fn __hits__(&self) -> Vec<usize> {
        self.hits.iter().map(|id| (*id).into()).collect()
    }

    pub fn avg_hit_confidence(&self) -> Option<f64> {
        self.avg_hit_confidence
    }

    pub fn avg_miss_confidence(&self) -> Option<f64> {
        self.avg_miss_confidence
    }

    pub fn combine(&self, rhs: Self) -> Self {
        FirstErrorDistance {
            hits: self.hits.union(&rhs.hits).cloned().collect(),
            misses: self.misses.union(&rhs.misses).cloned().collect(),
            avg_hit_confidence: combine_probs_options(
                self.avg_hit_confidence,
                rhs.avg_hit_confidence,
            ),
            avg_miss_confidence: combine_probs_options(
                self.avg_miss_confidence,
                rhs.avg_miss_confidence,
            ),
        }
    }
}

impl FirstErrorDistance {
    pub fn hits(&self) -> &HashSet<Id> {
        &self.hits
    }

    pub fn misses(&self) -> &HashSet<Id> {
        &self.misses
    }
}

pub fn compare<L: Language>(
    ground_truth: &RecExpr<L>,
    sample: &RecExpr<ProbabilisticLang<L>>,
) -> FirstErrorDistance {
    fn rec<LL: Language>(
        ground_truth: &RecExpr<LL>,
        gt_id: Id,
        sample: &RecExpr<ProbabilisticLang<LL>>,
        sample_id: Id,
    ) -> FirstErrorDistance {
        let gt_node = &ground_truth[gt_id];
        let sample_node = &sample[sample_id];
        if gt_node.matches(sample_node.inner()) {
            gt_node
                .children()
                .iter()
                .zip(sample_node.children().iter())
                .fold(
                    FirstErrorDistance {
                        hits: HashSet::from([sample_id]),
                        avg_hit_confidence: sample_node.prob(),
                        ..Default::default()
                    },
                    |acc, (gt_child, sample_child)| {
                        acc.combine(rec(ground_truth, *gt_child, sample, *sample_child))
                    },
                )
        } else {
            FirstErrorDistance {
                misses: HashSet::from([sample_id]),
                avg_miss_confidence: sample_node.prob(),
                ..Default::default()
            }
        }
    }
    rec(&ground_truth, ground_truth.root(), &sample, sample.root())
}

fn combine_probs_options(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    match (lhs, rhs) {
        (None, None) => None,
        (None, Some(x)) | (Some(x), None) => Some(x),
        (Some(x), Some(y)) => Some((x + y) / 2.0),
    }
}
