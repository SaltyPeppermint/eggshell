mod cost;
mod count;

use std::fmt::{Debug, Display};
use std::sync::atomic::{AtomicUsize, Ordering};

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use log::{debug, info, warn};
use rand::seq::IteratorRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use super::SampleError;
use super::{choices::ChoiceList, SampleConf};

pub use cost::CostWeighted;
pub use count::{CountWeightedGreedy, CountWeightedUniformly};

pub trait Strategy<'a, L, N>: Debug + Send + Sync
where
    L: Language + Display + Debug + Send + Sync + std::hash::Hash + 'a,
    N: Analysis<L> + Debug + 'a,
    N::Data: Debug + Sync,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        choices: &ChoiceList<L>,
    ) -> &'c L;

    fn egraph(&self) -> &'a EGraph<L, N>;

    #[expect(clippy::missing_errors_doc)]
    fn extractable(&self, id: Id) -> Result<(), SampleError>;

    #[expect(clippy::missing_errors_doc)]
    fn sample_expr(
        &self,
        rng: &mut ChaCha12Rng,
        root_eclass: &EClass<L, N::Data>,
    ) -> Result<RecExpr<L>, SampleError> {
        let egraph = self.egraph();

        let canonical_root_id = egraph.find(root_eclass.id);
        self.extractable(canonical_root_id)?;

        let mut choices = ChoiceList::from(canonical_root_id);

        while let Some(id) = choices.select_next_open(rng) {
            let eclass = &egraph[id];
            let pick = self.pick(rng, eclass, &choices);
            choices.fill_next(pick)?;
            if choices.len() > 10000 && choices.len() % 100 == 0 {
                warn!("Building very large sample with {} entries!", choices.len());
            }
        }
        let expr: RecExpr<L> = choices.try_into().expect("No open choices should be left");
        debug!(
            "Sampled expression of size {}: {} ",
            expr.as_ref().len(),
            expr
        );
        Ok(expr)
    }

    #[expect(clippy::missing_errors_doc)]
    fn sample_eclass(
        &self,
        rng: &mut ChaCha12Rng,
        conf: &SampleConf,
        root: Id,
    ) -> Result<HashSet<RecExpr<L>>, SampleError> {
        let root_eclass = &self.egraph()[root];

        let ranges = batch_ranges(conf);
        info!("Running sampling in {} batches", ranges.len());

        let counter = AtomicUsize::new(0);

        let samples = ranges
            .into_par_iter() // into_par_iter_here
            .enumerate()
            .map(|(range_id, (batch_start, batch_end))| {
                let mut inner_rng = rng.clone();
                inner_rng.set_stream((range_id + 1) as u64);

                let batch_samples = (batch_start..batch_end)
                    .map(|_| self.sample_expr(&mut inner_rng, root_eclass))
                    .collect::<Result<_, _>>()?;
                let len = batch_end - batch_start;
                let c = counter.fetch_add(len, Ordering::SeqCst) + len;
                debug!("Finished sampling batch {}", range_id + 1);
                debug!(
                    "Sampled {c} expressions from eclass {} in batch {range_id}",
                    root_eclass.id
                );

                Ok(batch_samples)
            })
            .try_reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })?;
        info!("Sampled {} expressions from {root}", samples.len());
        Ok(samples)
    }

    #[expect(clippy::missing_errors_doc)]
    fn sample_egraph(
        &self,
        rng: &mut ChaCha12Rng,
        conf: &SampleConf,
    ) -> Result<HashMap<Id, HashSet<RecExpr<L>>>, SampleError> {
        self.egraph()
            .classes()
            .choose_multiple(rng, conf.samples_per_egraph)
            .into_iter()
            .enumerate()
            .map(|(i, eclass)| {
                let samples = self.sample_eclass(rng, conf, eclass.id)?;
                info!("Sampled {i} expressions from eclass {}", eclass.id);
                Ok((eclass.id, samples))
            })
            .collect()
    }
}

fn batch_ranges(conf: &SampleConf) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut range_start = 0;

    while range_start + conf.batch_size < conf.samples_per_eclass {
        ranges.push((range_start, range_start + conf.batch_size));
        range_start += conf.batch_size;
    }

    ranges.push((range_start, conf.samples_per_eclass));
    ranges
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn batch_ranges_small() {
        let sample_conf = SampleConf::builder()
            .samples_per_eclass(2500)
            .batch_size(1000)
            .build();

        let ranges = batch_ranges(&sample_conf);
        assert_eq!(ranges, vec![(0, 1000), (1000, 2000), (2000, 2500)]);
    }
}
