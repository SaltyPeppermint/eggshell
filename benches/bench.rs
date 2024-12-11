use criterion::black_box as bb;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use egg::AstSize;
use egg::{EGraph, RecExpr, SymbolLang};
use rand::SeedableRng;

use eggshell::eqsat::{Eqsat, EqsatConf, StartMaterial};
use eggshell::sampling::strategy;
use eggshell::sampling::strategy::Strategy;
use eggshell::sampling::SampleConf;
use eggshell::sketch::extract;
use eggshell::sketch::Sketch;
use eggshell::trs::{Simple, TermRewriteSystem};
use rand_chacha::ChaCha12Rng;

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

    // c.bench_function("default extract", |b| {
    //     b.iter(|| extract::mutable::eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    // });
    // c.bench_function("recursive for_each extract", |b| {
    //     b.iter(|| {
    //         extract::recursive::for_each_eclass_extract(
    //             bb(&sketch),
    //             AstSize,
    //             bb(&egraph),
    //             bb(root_a),
    //         )
    //     })
    // });
    c.bench_function("recursive map extract", |b| {
        b.iter(|| extract::eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
}

fn sampling(c: &mut Criterion) {
    let start_expr = "(+ c (* (+ a b) 1))".parse().unwrap();
    let sample_conf = SampleConf::default();
    let eqsat_conf = EqsatConf::default();
    let rules = Simple::full_rules();
    let eqsat = Eqsat::new(StartMaterial::RecExprs(vec![start_expr]))
        .with_conf(eqsat_conf.clone())
        .run(&rules);

    let mut rng = ChaCha12Rng::seed_from_u64(sample_conf.rng_seed);
    let strategy = strategy::CostWeighted::new(eqsat.egraph(), AstSize, sample_conf.loop_limit);

    c.bench_function("sample simple", |b| {
        b.iter(|| strategy.sample_egraph(&mut rng, &sample_conf))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(1000);
    targets = extraction, sampling
);

criterion_main!(benches);
