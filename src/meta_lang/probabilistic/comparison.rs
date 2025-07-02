use std::ops::Add;

use egg::{Language, RecExpr};

use crate::meta_lang::{PartialLang, ProbabilisticLang};

pub struct FirstErrorDistance {
    misses: usize,
    avg_hit_confidence: Option<f64>,
    avg_miss_confidence: Option<f64>,
}

impl FirstErrorDistance {
    fn new(
        misses: usize,
        avg_hit_confidence: Option<f64>,
        avg_miss_confidence: Option<f64>,
    ) -> Self {
        Self {
            misses,
            avg_hit_confidence,
            avg_miss_confidence,
        }
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    pub fn avg_hit_confidence(&self) -> Option<f64> {
        self.avg_hit_confidence
    }

    pub fn avg_miss_confidence(&self) -> Option<f64> {
        self.avg_miss_confidence
    }

    pub fn compare<L: Language>(
        ground_truth: &RecExpr<L>,
        sample: &RecExpr<PartialLang<ProbabilisticLang<L>>>,
    ) -> FirstErrorDistance {
        fn rec<LL: Language>(
            ground_truth: &RecExpr<LL>,
            gt_node: &LL,
            sample: &RecExpr<PartialLang<ProbabilisticLang<LL>>>,
            sample_node: &PartialLang<ProbabilisticLang<LL>>,
        ) -> FirstErrorDistance {
            if let PartialLang::Finished(inner) = sample_node {
                if gt_node.matches(inner.inner()) {
                    gt_node
                        .children()
                        .iter()
                        .map(|c_id| &ground_truth[*c_id])
                        .zip(sample_node.children().iter().map(|c_id| &sample[*c_id]))
                        .fold(
                            FirstErrorDistance::new(0, inner.prob(), None),
                            |acc, (gt_child, sample_child)| {
                                acc + rec(ground_truth, gt_child, sample, sample_child)
                            },
                        )
                } else {
                    FirstErrorDistance::new(1, None, inner.prob())
                }
            } else {
                FirstErrorDistance::new(1, None, None)
            }
        }
        rec(
            &ground_truth,
            &ground_truth[ground_truth.root()],
            &sample,
            &sample[sample.root()],
        )
    }
}

impl Add<FirstErrorDistance> for FirstErrorDistance {
    type Output = FirstErrorDistance;

    fn add(self, rhs: FirstErrorDistance) -> Self::Output {
        FirstErrorDistance {
            misses: self.misses + rhs.misses,
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

fn combine_probs_options(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    Some((lhs? + rhs?) / 2.0)
}
