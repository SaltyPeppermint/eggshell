use criterion::black_box as bb;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use egg::AstSize;
use egg::{EGraph, RecExpr, SymbolLang};

use eggshell::eqsat::{Eqsat, EqsatConfBuilder, EqsatResult};
use eggshell::sampling::strategy;
use eggshell::sampling::strategy::Strategy;
use eggshell::sampling::SampleConfBuilder;
use eggshell::sketch::extract;
use eggshell::sketch::Sketch;
use eggshell::trs::Simple;
use eggshell::trs::Trs;
use rand::rngs::StdRng;
use rand::SeedableRng;

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

    c.bench_function("default extract", |b| {
        b.iter(|| extract::mutable::eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
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
        b.iter(|| extract::recursive::eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
}

fn sampling(c: &mut Criterion) {
    let term = "(+ c (* (+ a b) 1))";
    let seed: RecExpr<<Simple as Trs>::Language> = term.parse().unwrap();
    let sample_conf = SampleConfBuilder::new().build();
    let eqsat_conf = EqsatConfBuilder::new().build();
    let rules = Simple::full_rules();
    let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
        .with_conf(eqsat_conf.clone())
        .run(&rules);

    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);
    let mut strategy =
        strategy::CostWeighted::new(eqsat.egraph(), AstSize, &mut rng, sample_conf.loop_limit);

    c.bench_function("sample simple", |b| {
        b.iter(|| strategy.sample(&sample_conf))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(1000);
    targets = extraction, sampling
);

criterion_main!(benches);
