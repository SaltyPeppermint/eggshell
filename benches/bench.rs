use std::hint::black_box;

use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use egg::AstSize;
use egg::{EGraph, RecExpr, SymbolLang};
use eggshell::rewrite_system::simple::SimpleLang;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use eggshell::eqsat::Eqsat;
use eggshell::meta_lang::Sketch;
use eggshell::meta_lang::sketch;
use eggshell::rewrite_system::{RewriteSystem, Simple};
use eggshell::sampling::Sampler;
use eggshell::sampling::sampler;

fn extraction(c: &mut Criterion) {
    let sketch = "(contains (f ?))".parse::<Sketch<SymbolLang>>().unwrap();
    let a_expr = "(g (f (v x)))".parse::<RecExpr<SymbolLang>>().unwrap();
    let b_expr = "(h (g (f (u x))))".parse::<RecExpr<SymbolLang>>().unwrap();
    let mut egraph = EGraph::<SymbolLang, ()>::default();
    let root_a = egraph.add_expr(&a_expr);
    let root_b = egraph.add_expr(&b_expr);
    egraph.rebuild();
    egraph.union(root_a, root_b);
    egraph.rebuild();

    c.bench_function("recursive map extract", |b| {
        b.iter(|| {
            sketch::eclass_extract(
                black_box(&sketch),
                AstSize,
                black_box(&egraph),
                black_box(root_a),
            )
        })
    });
}

fn sampling(c: &mut Criterion) {
    let start_expr: RecExpr<SimpleLang> = "(+ c (* (+ a b) 1))".parse().unwrap();
    let rules = Simple::full_rules();
    let eqsat = Eqsat::new((&start_expr).into(), &rules).run();

    let mut rng = ChaCha12Rng::seed_from_u64(1024);
    let strategy = sampler::CostWeighted::new(eqsat.egraph(), AstSize);

    c.bench_function("sample simple", |b| {
        b.iter(|| strategy.sample_egraph(&mut rng, 1000, 4, 4))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(1000);
    targets = extraction, sampling
);

criterion_main!(benches);
