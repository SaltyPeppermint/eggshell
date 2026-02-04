//! Boltzmann Sampling for E-Graphs
//!
//! Provides methods for sampling terms from an e-graph with control over:
//! - Term size distribution (via Boltzmann parameter λ)
//! - Diversity (via structural hashing and deduplication)
//!
//! # Boltzmann Sampling
//!
//! Each term is weighted by `λ^size` where `λ ∈ (0, 1]`. Smaller λ values
//! bias toward smaller terms. The "critical" λ gives a target expected size.
//!
//! Iterative fixed-point computation handles cycles
//! correctly by converging to stable weights.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use super::graph::EGraph;
use super::ids::{EClassId, ExprChildId};
use super::nodes::Label;
use super::tree::TreeNode;

/// Trait for samplers that can draw terms from an e-graph.
pub trait Sampler: Sized {
    type Label: Label;

    /// Sample a single term, returning `None` if sampling fails.
    fn sample(&mut self) -> Option<TreeNode<Self::Label>>;

    /// Sample multiple terms.
    fn sample_many(&mut self, count: usize) -> Vec<TreeNode<Self::Label>> {
        (0..count).filter_map(|_| self.sample()).collect()
    }

    /// Convert this sampler into an iterator that yields samples.
    ///
    /// The iterator will call `sample()` on each `next()` and terminate
    /// when the sampler returns `None`. Use `.take(n)` to limit samples.
    ///
    /// # Example
    /// ```
    /// let sampler = FixpointSampler::new(&graph, &config, rng).unwrap();
    /// for tree in sampler.into_iter().take(100) {
    ///     println!("{:?}", tree);
    /// }
    /// ```
    fn into_sample_iter(self) -> SamplingIter<Self> {
        SamplingIter::new(self)
    }
}

/// Sampler using fixed-point iteration for weight computation.
///
/// This sampler handles cyclic e-graphs correctly by computing weights
/// via iterative convergence rather than recursion. Use this when your
/// e-graph may contain cycles.
pub struct FixpointSampler<'a, L: Label, R: Rng> {
    graph: &'a EGraph<L>,
    lambda: f64,
    weights: HashMap<EClassId, f64>,
    rng: R,
    max_depth: usize,
}

/// Configuration for the fixed-point sampler.
#[derive(Debug, Clone, bon::Builder)]
#[builder(derive(Clone, Debug))]
pub struct FixpointSamplerConfig {
    /// Convergence threshold for weight computation.
    #[builder(default = 1e-6)]
    pub epsilon: f64,
    /// Maximum iterations for weight convergence.
    #[builder(default = 1000)]
    pub max_iterations: usize,
    /// Maximum depth during sampling (prevents infinite loops on cycles).
    #[builder(default = 1000)]
    pub max_depth: usize,
}

impl FixpointSamplerConfig {
    /// Config for graphs with cycles - uses smaller lambda for faster convergence.
    #[must_use]
    pub fn for_cyclic() -> Self {
        Self::builder().max_depth(100).build()
    }
}

/// Find the lambda value that produces trees of approximately the target size.
///
/// Uses binary search over lambda, estimating average tree size at each point
/// by sampling. Larger lambda values produce larger trees (monotonic relationship).
///
/// # Arguments
/// * `graph` - The e-graph to sample from
/// * `target_size` - The desired average tree size (number of nodes)
/// * `config` - Configuration for the search
/// * `rng` - Random number generator (will be used to create seedable sub-rngs)
///
/// # Returns
/// * `Ok((lambda, actual_avg_size))` - The found lambda and the actual average size achieved
/// * `Err(FindLambdaError)` - If the search fails (e.g., target unreachable)
pub fn find_lambda_for_target_size<L: Label, R: Rng + SeedableRng>(
    graph: &EGraph<L>,
    target_size: usize,
    fixpoint_config: &FixpointSamplerConfig,
    rng: &mut R,
) -> Result<(f64, f64), FindLambdaError> {
    let mut lo = 0.001;
    let mut hi = 0.95; // Keep below 1 for convergence on cyclic graphs

    // First, check bounds to ensure target is achievable
    // Use a relative epsilon for convergence that scales with expected weight magnitude
    let size_at_min =
        estimate_avg_size(graph, fixpoint_config, lo, R::seed_from_u64(rng.next_u64()))?;
    let size_at_max =
        estimate_avg_size(graph, fixpoint_config, hi, R::seed_from_u64(rng.next_u64()))?;

    #[expect(clippy::cast_precision_loss)]
    let target = target_size as f64;

    if target < size_at_min {
        return Err(FindLambdaError::TargetTooSmall {
            target: target_size,
            min_achievable: size_at_min,
        });
    }
    if target > size_at_max {
        return Err(FindLambdaError::TargetTooLarge {
            target: target_size,
            max_achievable: size_at_max,
        });
    }

    // Binary search
    let mut best_lambda = f64::midpoint(lo, hi);
    let mut best_size = 0.0;

    while hi - lo > fixpoint_config.epsilon {
        let mid = f64::midpoint(lo, hi);
        let avg_size = estimate_avg_size(
            graph,
            fixpoint_config,
            mid,
            R::seed_from_u64(rng.next_u64()),
        )?;

        best_lambda = mid;
        best_size = avg_size;

        if avg_size < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((best_lambda, best_size))
}

/// Errors that can occur when finding lambda for a target size.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FindLambdaError {
    /// The target size is smaller than achievable with the minimum lambda.
    #[error("target size {target} is too small; minimum achievable is {min_achievable:.1}")]
    TargetTooSmall { target: usize, min_achievable: f64 },
    /// The target size is larger than achievable with the maximum lambda.
    #[error("target size {target} is too large; maximum achievable is {max_achievable:.1}")]
    TargetTooLarge { target: usize, max_achievable: f64 },
    /// The sampler failed to converge for a given lambda.
    #[error("sampler did not converge for lambda={lambda}")]
    SamplerDidNotConverge { lambda: f64 },
}

/// Estimate the average tree size for a given lambda by sampling.
fn estimate_avg_size<L: Label, R: Rng>(
    graph: &EGraph<L>,
    config: &FixpointSamplerConfig,
    lambda: f64,
    rng: R,
) -> Result<f64, FindLambdaError> {
    let samples = FixpointSampler::new(graph, lambda, config, rng)
        .ok_or(FindLambdaError::SamplerDidNotConverge { lambda })?
        .sample_many(1000);

    if samples.is_empty() {
        return Err(FindLambdaError::SamplerDidNotConverge { lambda });
    }

    #[expect(clippy::cast_precision_loss)]
    let avg = samples.iter().map(|t| t.size()).sum::<usize>() as f64 / samples.len() as f64;

    Ok(avg)
}

impl<'a, L: Label, R: Rng> FixpointSampler<'a, L, R> {
    /// Create a new fixed-point sampler.
    ///
    /// Computes weights via fixed-point iteration until convergence.
    ///
    /// # Returns
    /// `None` if weight computation does not converge, otherwise the sampler.
    ///
    /// # Panics
    /// Panics if `lambda` is not in the range (0, 1].
    pub fn new(
        graph: &'a EGraph<L>,
        lambda: f64,
        config: &FixpointSamplerConfig,
        rng: R,
    ) -> Option<Self> {
        assert!(lambda > 0.0 && lambda <= 1.0, "λ must be in (0, 1]");

        // Collect canonical class IDs and initialize weights
        let mut weights = graph
            .class_ids()
            .map(|id| (graph.canonicalize(id), lambda))
            .collect::<HashMap<_, _>>();
        let class_ids = weights.keys().copied().collect::<Vec<_>>();

        for _ in 0..config.max_iterations {
            let mut max_delta: f64 = 0.0;

            for &id in &class_ids {
                let new_weight: f64 = graph
                    .class(id)
                    .nodes()
                    .iter()
                    .map(|node| {
                        let child_product: f64 = node
                            .children()
                            .iter()
                            .map(|child| match child {
                                ExprChildId::EClass(eid) => weights[&graph.canonicalize(*eid)],
                                ExprChildId::Nat(_) | ExprChildId::Data(_) => lambda,
                            })
                            .product();
                        lambda * child_product
                    })
                    .sum();

                let old_weight = weights[&id];
                max_delta = max_delta.max((new_weight - old_weight).abs());
                weights.insert(id, new_weight);
            }

            if max_delta < config.epsilon {
                return Some(Self {
                    graph,
                    lambda,
                    weights,
                    rng,
                    max_depth: config.max_depth,
                });
            }
        }
        None
    }

    /// Compute weights for each node choice at an e-class.
    fn node_weights(&self, id: EClassId) -> Vec<f64> {
        let canonical = self.graph.canonicalize(id);
        self.graph
            .class(canonical)
            .nodes()
            .iter()
            .map(|node| {
                let child_product: f64 = node
                    .children()
                    .iter()
                    .map(|child| match child {
                        ExprChildId::EClass(eid) => {
                            let c = self.graph.canonicalize(*eid);
                            self.weights.get(&c).copied().unwrap_or(0.0)
                        }
                        ExprChildId::Nat(_) | ExprChildId::Data(_) => self.lambda,
                    })
                    .product();
                self.lambda * child_product
            })
            .collect()
    }

    /// Sample a term rooted at the given e-class.
    fn sample_from(&mut self, id: EClassId, depth: usize) -> Option<TreeNode<L>> {
        if depth >= self.max_depth {
            return None;
        }

        let canonical = self.graph.canonicalize(id);
        let node_weights = self.node_weights(canonical);
        let dist = WeightedIndex::new(&node_weights).ok()?;
        let chosen_idx = self.rng.sample(dist);
        let chosen_node = &self.graph.class(canonical).nodes()[chosen_idx];

        chosen_node
            .children()
            .iter()
            .map(|child| self.sample_child(child, depth + 1))
            .collect::<Option<Vec<_>>>()
            .map(|children| TreeNode::new(chosen_node.label().clone(), children))
    }

    /// Sample a child, dispatching on child type.
    fn sample_child(&mut self, child: &ExprChildId, depth: usize) -> Option<TreeNode<L>> {
        match child {
            ExprChildId::EClass(eid) => self.sample_from(*eid, depth),
            ExprChildId::Nat(nid) => Some(TreeNode::from_nat(self.graph, *nid)),
            ExprChildId::Data(did) => Some(TreeNode::from_data(self.graph, *did)),
        }
    }
}

impl<L: Label, R: Rng> Sampler for FixpointSampler<'_, L, R> {
    type Label = L;

    fn sample(&mut self) -> Option<TreeNode<L>> {
        self.sample_from(self.graph.root(), 0)
    }
}

/// Configuration for diverse sampling.
#[derive(Debug, Clone, bon::Builder)]
pub struct DiverseSamplerConfig {
    /// Maximum attempts to find a novel sample before giving up.
    #[builder(default = 100)]
    pub max_attempts_per_sample: usize,
    /// Minimum novelty ratio (0, 1]. Sample accepted if this fraction of features are new.
    #[builder(default = 0.3)]
    pub min_novelty_ratio: f64,
}

/// Sampler that produces diverse terms using structural deduplication.
/// Generic over any underlying sampler.
pub struct DiverseSampler<S: Sampler> {
    sampler: S,
    config: DiverseSamplerConfig,
    seen_hashes: HashSet<u64>,
    seen_features: HashSet<(S::Label, usize, S::Label)>,
}

impl<S: Sampler> DiverseSampler<S> {
    /// Create a new diverse sampler wrapping an existing sampler.
    pub fn new(sampler: S, config: DiverseSamplerConfig) -> Self {
        Self {
            sampler,
            config,
            seen_hashes: HashSet::new(),
            seen_features: HashSet::new(),
        }
    }

    /// Check if a term is novel enough to accept.
    #[expect(clippy::type_complexity)]
    fn is_novel(
        &self,
        term: &TreeNode<S::Label>,
    ) -> (bool, u64, HashSet<(S::Label, usize, S::Label)>) {
        let hash = structural_hash(term);
        let features = extract_features(term);

        // Novel if we've never seen this exact structure
        if !self.seen_hashes.contains(&hash) {
            return (true, hash, features);
        }

        // Otherwise check feature novelty ratio
        let novel_count = features
            .iter()
            .filter(|f| !self.seen_features.contains(*f))
            .count();

        #[expect(clippy::cast_precision_loss)]
        let novelty_ratio = if features.is_empty() {
            0.0
        } else {
            novel_count as f64 / features.len() as f64
        };

        (
            novelty_ratio >= self.config.min_novelty_ratio,
            hash,
            features,
        )
    }

    /// Accept a term, updating seen hashes and features.
    fn accept(&mut self, hash: u64, features: HashSet<(S::Label, usize, S::Label)>) {
        self.seen_hashes.insert(hash);
        self.seen_features.extend(features);
    }

    /// Reset the diversity tracking state.
    pub fn reset(&mut self) {
        self.seen_hashes.clear();
        self.seen_features.clear();
    }

    pub fn seen_hashes(&self) -> &HashSet<u64> {
        &self.seen_hashes
    }

    pub fn seen_features(&self) -> &HashSet<(S::Label, usize, S::Label)> {
        &self.seen_features
    }
}

impl<S: Sampler> Sampler for DiverseSampler<S> {
    type Label = S::Label;

    fn sample(&mut self) -> Option<TreeNode<Self::Label>> {
        let this = &mut *self;
        for _ in 0..this.config.max_attempts_per_sample {
            let term = this.sampler.sample()?;
            let (is_novel, hash, features) = this.is_novel(&term);
            if is_novel {
                this.accept(hash, features);
                return Some(term);
            }
        }
        None
    }
}

/// Compute a structural hash of a tree for diversity checking.
/// Trees with the same structure and labels will have the same hash.
#[must_use]
pub fn structural_hash<L: Label>(tree: &TreeNode<L>) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_tree_rec(tree, &mut hasher);
    hasher.finish()
}

fn hash_tree_rec<L: Label, H: Hasher>(tree: &TreeNode<L>, hasher: &mut H) {
    tree.label().hash(hasher);
    tree.children().len().hash(hasher);
    for child in tree.children() {
        hash_tree_rec(child, hasher);
    }
}

/// Extract structural features from a tree for diversity measurement.
/// Returns bigrams of `(parent_label, child_index, child_label)`.
#[must_use]
pub fn extract_features<L: Label>(tree: &TreeNode<L>) -> HashSet<(L, usize, L)> {
    let mut features = HashSet::new();
    collect_features(tree, &mut features);
    features
}

fn collect_features<L: Label>(tree: &TreeNode<L>, features: &mut HashSet<(L, usize, L)>) {
    let parent = tree.label().clone();
    for (i, child) in tree.children().iter().enumerate() {
        features.insert((parent.clone(), i, child.label().clone()));
        collect_features(child, features);
    }
}

/// Iterator adapter that yields samples from any `Sampler`.
///
/// This iterator wraps a sampler and calls `sample()` on each `next()`.
/// The iterator terminates when the sampler returns `None`, or you can
/// use methods like `.take(n)` to limit the number of samples.
///
/// # Example
/// ```ignore
/// let sampler = FixpointSampler::new(&graph, &config, rng).unwrap();
/// let iter = SamplingIter::new(sampler);
/// for tree in iter.take(100) {
///     println!("{:?}", tree);
/// }
/// ```
pub struct SamplingIter<S: Sampler> {
    sampler: S,
}

impl<S: Sampler> SamplingIter<S> {
    /// Create a new sampling iterator from a sampler.
    pub fn new(sampler: S) -> Self {
        Self { sampler }
    }

    /// Consume the iterator and return the underlying sampler.
    pub fn into_inner(self) -> S {
        self.sampler
    }

    /// Get a reference to the underlying sampler.
    pub fn sampler(&self) -> &S {
        &self.sampler
    }

    /// Get a mutable reference to the underlying sampler.
    pub fn sampler_mut(&mut self) -> &mut S {
        &mut self.sampler
    }
}

impl<S: Sampler> Iterator for SamplingIter<S> {
    type Item = TreeNode<S::Label>;

    fn next(&mut self) -> Option<Self::Item> {
        self.sampler.sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::graph::EClass;
    use crate::distance::ids::NatId;
    use crate::distance::ids::TypeChildId;
    use crate::distance::nodes::{ENode, NatNode};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn dummy_ty() -> TypeChildId {
        TypeChildId::Nat(NatId::new(0))
    }

    fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
        let mut map = HashMap::new();
        map.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
        map
    }

    fn eid(i: usize) -> ExprChildId {
        ExprChildId::EClass(EClassId::new(i))
    }

    fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
        classes
            .into_iter()
            .enumerate()
            .map(|(i, c)| (EClassId::new(i), c))
            .collect()
    }

    fn cyclic_graph() -> EGraph<String> {
        // Class 0: "f" with child Class 0 (cycle!), or leaf "x"
        // This represents: x, f(x), f(f(x)), f(f(f(x))), ...
        EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("f".to_owned(), vec![eid(0)]), // cycle back to self
                    ENode::leaf("x".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        )
    }

    #[test]
    fn fixpoint_handles_cycles() {
        let graph = cyclic_graph();
        let config = FixpointSamplerConfig::for_cyclic();
        let rng = StdRng::seed_from_u64(42);

        let mut sampler = FixpointSampler::new(&graph, 0.5, &config, rng)
            .expect("Should converge with λ < 1 on cyclic graph");

        // Should be able to sample without infinite loop
        let terms = sampler.sample_many(100);
        assert!(!terms.is_empty(), "Should produce some terms");

        // With small lambda, most terms should be small
        let small_terms = terms.iter().filter(|t| t.label() == "x").count();
        assert!(
            small_terms > 30,
            "With λ=0.5, should prefer leaf 'x', got {small_terms}/100"
        );
    }

    #[test]
    fn fixpoint_weights_converge() {
        let graph = cyclic_graph();

        // With λ = 0.5, the weight equation for the cyclic class is:
        // W = λ × W + λ  (from "f" with child + "x" leaf)
        // W = 0.5W + 0.5
        // 0.5W = 0.5
        // W = 1.0
        //
        // This means P(x) = 0.5/1.0 = 50%, P(f(...)) = 50%
        let config = FixpointSamplerConfig::builder().max_depth(100).build();
        let rng = StdRng::seed_from_u64(42);
        let mut sampler = FixpointSampler::new(&graph, 0.5, &config, rng).unwrap();

        // Verify the probability distribution matches theory
        let leaf_count = sampler
            .sample_many(1000)
            .iter()
            .filter(|t| t.label() == "x")
            .count();

        // Should be close to 50% (allowing for variance)
        assert!(
            (400..600).contains(&leaf_count),
            "Expected ~50% leaves, got {leaf_count}/1000"
        );
    }

    #[test]
    fn sampling_iter_yields_samples() {
        let graph = cyclic_graph();
        let config = FixpointSamplerConfig::for_cyclic();
        let rng = StdRng::seed_from_u64(42);

        // Use the iterator interface
        let samples = FixpointSampler::new(&graph, 0.5, &config, rng)
            .expect("Should converge with λ < 1 on cyclic graph")
            .into_sample_iter()
            .take(50)
            .collect::<Vec<_>>();

        assert_eq!(samples.len(), 50, "Should yield exactly 50 samples");

        // All samples should have valid root labels
        for sample in &samples {
            assert!(
                sample.label() == "f" || sample.label() == "x",
                "Unexpected root label: {}",
                sample.label()
            );
        }
    }

    #[test]
    fn sampling_iter_can_access_sampler() {
        let graph = cyclic_graph();
        let config = FixpointSamplerConfig::for_cyclic();
        let rng = StdRng::seed_from_u64(42);

        let sampler = FixpointSampler::new(&graph, 0.5, &config, rng).unwrap();
        let mut iter = SamplingIter::new(sampler);

        // Take some samples
        let _ = iter.next();
        let _ = iter.next();

        // We can still access the sampler through the iterator
        let _sampler_ref = iter.sampler();

        // And recover the sampler when done
        let _sampler = iter.into_inner();
    }

    #[test]
    fn find_lambda_for_target_size_works() {
        let graph = cyclic_graph();

        let mut rng = StdRng::seed_from_u64(42);
        let config = FixpointSamplerConfig::builder().build();

        // Target size of 2 (e.g., f(x) has 2 nodes)
        let result = find_lambda_for_target_size(&graph, 2, &config, &mut rng);
        assert!(
            result.is_ok(),
            "Should find a lambda for target size 2: {:?}",
            result.err()
        );

        let (lambda, avg_size) = result.unwrap();
        assert!(lambda > 0.0 && lambda < 1.0, "Lambda should be in (0, 1)");
        // Allow some variance in the achieved size
        assert!(
            (1.5..4.0).contains(&avg_size),
            "Average size {avg_size} should be roughly near target 2"
        );
    }
}
