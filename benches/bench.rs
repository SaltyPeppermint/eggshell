use criterion::black_box as bb;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use egg::{AstSize, EGraph, RecExpr, SymbolLang};

use eggshell::sketch::Sketch;
use eggshell::sketch::{extract, recursive};

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
        b.iter(|| extract::eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
    c.bench_function("recursive for_each extract", |b| {
        b.iter(|| recursive::for_each_eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
    c.bench_function("recursive map extract", |b| {
        b.iter(|| recursive::map_eclass_extract(bb(&sketch), AstSize, bb(&egraph), bb(root_a)))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(1000);
    targets = extraction
);
criterion_main!(benches);
