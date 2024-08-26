use criterion::black_box as bb;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use egg::{EGraph, RecExpr, SymbolLang};

use eggshell::eqsat::{Eqsat, EqsatConfBuilder, EqsatResult};
use eggshell::sampling;
use eggshell::sampling::SampleConfBuilder;
use eggshell::sketch::Sketch;
use eggshell::sketch::{extract, recursive};
use eggshell::trs::simple;
use eggshell::trs::simple::SimpleLang;
use eggshell::trs::Simple;
use eggshell::trs::Trs;
use eggshell::utils::AstSize2;
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
        b.iter(|| extract::eclass_extract(bb(&sketch), AstSize2, bb(&egraph), bb(root_a)))
    });
    c.bench_function("recursive for_each extract", |b| {
        b.iter(|| {
            recursive::for_each_eclass_extract(bb(&sketch), AstSize2, bb(&egraph), bb(root_a))
        })
    });
    c.bench_function("recursive map extract", |b| {
        b.iter(|| recursive::map_eclass_extract(bb(&sketch), AstSize2, bb(&egraph), bb(root_a)))
    });
}

fn sampling(c: &mut Criterion) {
    let term = "(+ c (* (+ a b) 1))";
    let seed: RecExpr<SimpleLang> = term.parse().unwrap();
    let sample_conf = SampleConfBuilder::new().build();
    let eqsat_conf = EqsatConfBuilder::new().build();

    let rules = Simple::rules(&simple::Ruleset::Full);
    let eqsat: EqsatResult<Simple> = Eqsat::new(vec![seed])
        .with_conf(eqsat_conf.clone())
        .run(&rules);

    let mut rng = StdRng::seed_from_u64(sample_conf.rng_seed);

    c.bench_function("sample simple", |b| {
        b.iter(|| sampling::sample(bb(eqsat.egraph()), &sample_conf, &mut rng))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(1000);
    targets = extraction, sampling
);

criterion_main!(benches);
