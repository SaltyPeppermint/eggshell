mod cost;
mod count;
mod simple;

use std::{
    fmt::{Debug, Display},
    ops::Range,
};

use egg::{Analysis, EClass, EGraph, Id, Language, RecExpr};
use hashbrown::{HashMap, HashSet};
use log::{debug, info, warn};
use rand::seq::IteratorRandom;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use super::{PartialRecExpr, SampleError};

pub use cost::CostWeighted;
pub use count::CountUniformly;
pub use simple::Greedy;

pub trait Sampler<'a, L, N>: Debug + Send + Sync
where
    L: Language + Display + Send + Sync + 'a,
    N: Analysis<L> + Debug + 'a,
    N::Data: Sync,
{
    fn pick<'c: 'a>(
        &self,
        rng: &mut ChaCha12Rng,
        eclass: &'c EClass<L, N::Data>,
        size_limit: usize,
        partial_rec_expr: &PartialRecExpr<L>,
    ) -> &'c L;

    fn egraph(&self) -> &'a EGraph<L, N>;

    fn extractable(&self, id: Id, size_limit: usize) -> bool;

    #[expect(clippy::missing_errors_doc)]
    fn sample_expr(
        &self,
        rng: &mut ChaCha12Rng,
        root_eclass: &EClass<L, N::Data>,
        size_limit: usize,
    ) -> Result<RecExpr<L>, SampleError> {
        let egraph = self.egraph();
        let canonical_root_id = egraph.find(root_eclass.id);

        let mut partial_rec_expr = PartialRecExpr::from(canonical_root_id);

        while let Some((id, key)) = partial_rec_expr.select_next_open(rng) {
            let eclass = &egraph[id];
            let pick = self.pick(rng, eclass, size_limit, &partial_rec_expr);
            partial_rec_expr.fill_next(key, pick)?;
            if partial_rec_expr.len() > 10000 && partial_rec_expr.len() % 100 == 0 {
                warn!(
                    "Building very large sample with {} entries!",
                    partial_rec_expr.len()
                );
            }
        }
        let expr: RecExpr<L> = partial_rec_expr
            .try_into()
            .expect("No open choices should be left");
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
        rng: &ChaCha12Rng,
        n_samples: usize,
        root: Id,
        size_limit: usize,
        parallelism: usize,
    ) -> Result<HashSet<RecExpr<L>>, SampleError> {
        let root_eclass = &self.egraph()[root];

        let ranges = batch_ranges(n_samples, parallelism);
        info!("Running sampling in {} batches", ranges.len());
        if !self.extractable(root, size_limit) {
            return Err(SampleError::SizeLimit(size_limit));
        }

        let samples = ranges
            .into_par_iter() // into_par_iter_here
            .enumerate()
            .map(|(range_id, range)| {
                let mut inner_rng = rng.clone();
                inner_rng.set_stream((range_id + 1) as u64);
                range
                    .map(|_| self.sample_expr(&mut inner_rng, root_eclass, size_limit))
                    .collect::<Result<_, _>>()
            })
            .try_reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })?;
        let n_samples = samples.len();
        info!("Sampled {n_samples} expressions from eclass {root} with size_limit {size_limit}");
        Ok(samples)
    }

    #[expect(clippy::missing_errors_doc)]
    fn sample_egraph(
        &self,
        rng: &mut ChaCha12Rng,
        n_samples: usize,
        size_limit: usize,
        parallelism: usize,
    ) -> Result<HashMap<Id, HashSet<RecExpr<L>>>, SampleError> {
        self.egraph()
            .classes()
            .choose_multiple(rng, n_samples)
            .into_iter()
            .enumerate()
            .map(|(i, eclass)| {
                let samples =
                    self.sample_eclass(rng, n_samples, eclass.id, size_limit, parallelism)?;
                info!("Sampled {i} expressions from eclass {}", eclass.id);
                Ok((eclass.id, samples))
            })
            .collect()
    }
}

fn batch_ranges(total_samples: usize, parallelism: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let batch_size = total_samples / parallelism;
    let mut range_start = 0;

    loop {
        let range_end = range_start + batch_size;
        if range_end >= total_samples {
            ranges.push(range_start..total_samples);
            break;
        }
        ranges.push(range_start..range_end);
        range_start = range_end;
    }
    ranges
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn batch_ranges_2001() {
        let ranges = batch_ranges(2001, 4);
        assert_eq!(
            ranges,
            vec![
                (0..500),
                (500..1000),
                (1000..1500),
                (1500..2000),
                (2000..2001)
            ]
        );
    }

    #[test]
    fn batch_ranges_1999() {
        let ranges = batch_ranges(1999, 4);
        assert_eq!(
            ranges,
            vec![
                (0..499),
                (499..998),
                (998..1497),
                (1497..1996),
                (1996..1999)
            ]
        );
    }

    #[test]
    fn batch_ranges_2000() {
        let ranges = batch_ranges(2000, 4);
        assert_eq!(
            ranges,
            vec![(0..500), (500..1000), (1000..1500), (1500..2000)]
        );
    }

    #[test]
    fn batch_ranges_20000() {
        let ranges = batch_ranges(20000, 4);
        assert_eq!(
            ranges,
            vec![0..5000, 5000..10000, 10000..15000, 15000..20000]
        );
    }
}
