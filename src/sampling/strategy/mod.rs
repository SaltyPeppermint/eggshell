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
pub use count::{CountLutWeighted, CountWeighted, CountWeightedUniformly};

const BATCH_SIZE: usize = 1000;

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

    /// .
    ///
    /// # Errors
    ///
    /// This function will return an error if .
    fn extractable(&self, id: Id) -> Result<(), SampleError>;

    /// .
    ///
    /// # Panics
    ///
    /// Panics if .
    ///
    /// # Errors
    ///
    /// This function will return an error if something goes wrong during sampling.
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
            let eclass_id = egraph.find(id);
            let eclass = &egraph[eclass_id];
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
                let mut h = HashSet::new();
                let mut inner_rng = rng.clone();
                inner_rng.set_stream((range_id + 1) as u64);
                for _ in batch_start..=batch_end {
                    let sample = self.sample_expr(&mut inner_rng, root_eclass)?;
                    h.insert(sample);
                }
                let c = counter.fetch_add(BATCH_SIZE, Ordering::SeqCst) + BATCH_SIZE;
                info!("Finsihed sampling batch {range_id}");
                info!("Sampled {c} expressions from eclass {}", root_eclass.id);

                Ok(h)
            })
            // .try_fold(HashSet::new(), |mut a, b| {
            //     a.extend(b?);
            //     Ok(a)
            // })?;
            .try_reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })?;

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
    let mut start = 0;

    loop {
        let end = start + BATCH_SIZE;
        if end < conf.samples_per_eclass {
            ranges.push((start, end));
        } else {
            ranges.push((start, conf.samples_per_eclass));
            break;
        }
        start = end + 1;
    }
    ranges
}
