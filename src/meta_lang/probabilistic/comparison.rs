use egg::{Id, Language, RecExpr};
use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use serde::Serialize;

use crate::{
    meta_lang::ProbabilisticLang,
    rewrite_system::{LangExtras, SymbolType},
};

#[gen_stub_pyclass]
#[pyclass(module = "eggshell", frozen)]
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct FirstErrorDistance {
    hits: HashMap<Id, Option<f64>>,
    misses: HashMap<Id, Option<f64>>,
}

impl From<&FirstErrorDistance> for FirstErrorDistance {
    fn from(value: &FirstErrorDistance) -> Self {
        value.to_owned()
    }
}

impl Default for FirstErrorDistance {
    fn default() -> Self {
        Self {
            hits: HashMap::new(),
            misses: HashMap::new(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FirstErrorDistance {
    #[new]
    fn new_py() -> FirstErrorDistance {
        FirstErrorDistance::default()
    }

    pub fn hit_ids(&self) -> Vec<usize> {
        self.hits.iter().map(|hit| (*hit.0).into()).collect()
    }

    fn hit_probabilities(&self) -> Vec<Option<f64>> {
        self.hits.iter().map(|hit| *hit.1).collect()
    }

    #[getter(n_hits)]
    pub fn n_hits(&self) -> usize {
        self.hits.len()
    }

    pub fn miss_ids(&self) -> Vec<usize> {
        self.misses.iter().map(|miss| (*miss.0).into()).collect()
    }

    pub fn miss_probabilities(&self) -> Vec<Option<f64>> {
        self.misses.iter().map(|miss| *miss.1).collect()
    }

    #[getter(n_misses)]
    pub fn n_misses(&self) -> usize {
        self.misses.len()
    }
}

impl FirstErrorDistance {
    pub fn hits(&self) -> &HashMap<Id, Option<f64>> {
        &self.hits
    }

    pub fn misses(&self) -> &HashMap<Id, Option<f64>> {
        &self.misses
    }

    pub fn combine<T: Into<FirstErrorDistance>>(&mut self, rhs: T) {
        let o: FirstErrorDistance = rhs.into();
        self.hits.extend(o.hits);
        self.misses.extend(o.misses);
    }
}

impl<T: Into<FirstErrorDistance>> Extend<T> for FirstErrorDistance {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for el in iter {
            self.combine(el)
        }
    }
}

pub fn compare<L: Language + LangExtras>(
    ground_truth: &RecExpr<L>,
    sample: &RecExpr<ProbabilisticLang<L>>,
) -> FirstErrorDistance {
    fn rec<LL: Language + LangExtras>(
        ground_truth: &RecExpr<LL>,
        gt_id: Id,
        sample: &RecExpr<ProbabilisticLang<LL>>,
        sample_id: Id,
    ) -> FirstErrorDistance {
        let gt_node = &ground_truth[gt_id];
        let sample_node = &sample[sample_id];
        if gt_node.matches(sample_node.inner())
            || (matches!(gt_node.symbol_info().symbol_type(), SymbolType::Variable(_))
                && matches!(
                    sample_node.symbol_info().symbol_type(),
                    SymbolType::Variable(_)
                ))
        {
            gt_node
                .children()
                .iter()
                .zip(sample_node.children().iter())
                .fold(
                    FirstErrorDistance {
                        hits: HashMap::from([(sample_id, sample_node.prob())]),
                        ..Default::default()
                    },
                    |mut acc, (gt_child, sample_child)| {
                        acc.combine(rec(ground_truth, *gt_child, sample, *sample_child));
                        acc
                    },
                )
        } else {
            FirstErrorDistance {
                misses: HashMap::from([(sample_id, sample_node.prob())]),
                ..Default::default()
            }
        }
    }
    rec(&ground_truth, ground_truth.root(), &sample, sample.root())
}

// fn combine_probs_options(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
//     match (lhs, rhs) {
//         (None, None) => None,
//         (None, Some(x)) | (Some(x), None) => Some(x),
//         (Some(x), Some(y)) => Some((x + y) / 2.0),
//     }
// }
