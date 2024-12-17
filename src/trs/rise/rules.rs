use egg::{rewrite, Applier, Id, Pattern, PatternAst, Subst, Symbol, Var};
use hashbrown::HashMap;

use super::{substitute, EGraph, Rewrite, RiseAnalysis, RiseLang};

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

fn is_not_same_var(v1: Var, v2: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph.find(subst[v1]) != egraph.find(subst[v2])
}

fn contains_ident(v1: Var, v2: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph[subst[v1]].data.free.contains(&subst[v2])
}

fn neg(f: impl Fn(&mut EGraph, Id, &Subst) -> bool) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, id, subst| !f(egraph, id, subst)
}

fn is_const(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| {
        let e: &[RiseLang] = egraph[subst[v]].data.beta_extract.as_ref();
        (e.len() == 1) && matches!(e[0], RiseLang::Symbol(_) | RiseLang::Number(_))
    }
}

pub fn rules(use_explicit_subs: bool) -> HashMap<Symbol, Rewrite> {
    let common = vec![
        // Comment out to deal with unexpected new symbols
        // algorithmic
        rewrite!("map-fusion";
            "(app (app map ?f) (app (app map ?g) ?arg))" =>
            { with_fresh_var("?mfu", "(app (app map (lam ?mfu (app ?f (app ?g (var ?mfu))))) ?arg)") }),
        rewrite!("map-fission";
            "(app map (lam ?x (app ?f ?gx)))" =>
            { with_fresh_var("?mfi", "(lam ?mfi (app (app map ?f) (app (app map (lam ?x ?gx)) (var ?mfi))))") }
            // TODO: if conditions should be recursive filters?
            if neg(contains_ident(var("?f"), var("?x")))),
        rewrite!("map-fusion-then";
            "(>> (app map ?f) (app map ?g))" => "(app map (>> ?f ?g))"),
        rewrite!("map-fission-then";
            "(app map (>> ?f ?g))" => "(>> (app map ?f) (app map ?g))"),
        rewrite!("then-assoc-1"; "(>> ?f (>> ?g ?h))" => "(>> (>> ?f ?g) ?h)"),
        rewrite!("then-assoc-2"; "(>> (>> ?f ?g) ?h)" => "(>> ?f (>> ?g ?h))"),
        rewrite!("transpose-pair-after-then";
                 "?f" => "(>> (>> ?f transpose) transpose)"),
        rewrite!("map-map-f-before-transpose-then";
                 "(>> (app map (app map ?f)) transpose)" =>
                 "(>> transpose (app map (app map ?f)))"),
        rewrite!("split-join-then";
                 "(app map ?f)" => "(>> (>> split (app map (app map ?f))) join)"),
        // rewrite!("split-join-then";
        //         "(app map ?f)" => "(>> split (>> (app map (app map ?f)) join))"),

        // reductions
        rewrite!("eta"; "(lam ?v (app ?f (var ?v)))" => "?f"
            // TODO: if conditions should be recursive filters?
            if neg(contains_ident(var("?f"), var("?v")))),
        rewrite!("remove-transpose-pair";
            "(app transpose (app transpose ?x))" => "?x"),
        // movement
        rewrite!("map-slide-before-transpose";
            "(app transpose (app (app map (app (app slide ?sz) ?sp)) ?y))" =>
            "(app (app map transpose) (app (app (app slide ?sz) ?sp) (app transpose ?y)))"),
        rewrite!("map-split-before-transpose";
            "(app transpose (app (app map (app split ?n)) ?y))" =>
            "(app (app map transpose) (app (app split ?n) (app transpose ?y)))"),
        rewrite!("slide-before-map-map-f";
            "(app (app map (app map ?f)) (app (app (app slide ?sz) ?sp) ?y))" =>
            "(app (app (app slide ?sz) ?sp) (app (app map ?f) ?y))"),
        rewrite!("split-before-map-map-f";
            "(app (app map (app map ?f)) (app (app split ?n) ?y))" =>
            "(app (app split ?n) (app (app map ?f) ?y))"),
        rewrite!("slide-before-map";
            "(app (app (app slide ?sz) ?sp) (app (app map ?f) ?y))" =>
            "(app (app map (app map ?f)) (app (app (app slide ?sz) ?sp) ?y))"),
        // lowering
        rewrite!("reduce-seq-unroll"; "reduce" => "reduceSeqUnroll"),
        rewrite!("map-seq"; "map" => "mapSeq"),
        rewrite!("iterate-stream"; "map" => "iterateStream"),
        rewrite!("to-mem-after-map-seq";
            "(app (app mapSeq ?f) ?x)" => "(app toMem (app (app mapSeq ?f) ?x))"),
        rewrite!("rotate-values-simplified";
            "(app (app slide ?sz) 1)" => "(app rotateValues ?sz)"),
        // domain-specific
        // mulT = (lam x (app (app mul (app fst (var x))) (app snd (var x))))
        rewrite!("separate-dot-hv-simplified";
            "(app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
             (app (app zip (app join weights2d)) (app join ?nbh))))" =>
            { with_fresh_var("?sdhv", "(app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
                (app (app zip weightsV) (app (app map (lam ?sdhv (app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
                (app (app zip weightsH) (var ?sdhv)))))) ?nbh))))") }),
        rewrite!("separate-dot-vh-simplified";
            "(app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
             (app (app zip (app join weights2d)) (app join ?nbh))))" =>
            { with_fresh_var("?sdvh", "(app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
                (app (app zip weightsH) (app (app map (lam ?sdvh (app (app (app reduce add) 0) (app (app map (lam ?x (app (app mul (app fst (var ?x))) (app snd (var ?x)))))
                (app (app zip weightsV) (var ?sdvh)))))) (app transpose ?nbh)))))") }),
    ];
    let extraction_substitution = vec![rewrite!("beta"; "(app (lam ?v ?body) ?e)" =>
            { BetaExtractApplier { v: var("?v"), e: var("?e"), body: var("?body") } })];
    let explicit_substitution = vec![
        rewrite!("beta"; "(app (lam ?v ?body) ?e)" => "(let ?v ?e ?body)"),
        // explicit substitution
        rewrite!("let-app"; "(let ?v ?e (app ?a ?b))" => "(app (let ?v ?e ?a) (let ?v ?e ?b))"),
        rewrite!("let-var-same"; "(let ?v1 ?e (var ?v1))" => "?e"),
        rewrite!("let-var-diff"; "(let ?v1 ?e (var ?v2))" => "(var ?v2)"
            if is_not_same_var(var("?v1"), var("?v2"))),
        rewrite!("let-lam-same"; "(let ?v1 ?e (lam ?v1 ?body))" => "(lam ?v1 ?body)"),
        rewrite!("let-lam-diff"; "(let ?v1 ?e (lam ?v2 ?body))" =>
            { CaptureAvoid {
                fresh: var("?fresh"), v2: var("?v2"), e: var("?e"),
                if_not_free: "(lam ?v2 (let ?v1 ?e ?body))".parse().unwrap(),
                if_free: "(lam ?fresh (let ?v1 ?e (let ?v2 (var ?fresh) ?body)))".parse().unwrap(),
            }}
            if is_not_same_var(var("?v1"), var("?v2"))),
        rewrite!("let-const"; "(let ?v ?e ?c)" => "?c" if is_const(var("?c"))),
    ];
    let mut map: HashMap<Symbol, _> = common.into_iter().map(|r| (r.name, r)).collect();
    if use_explicit_subs {
        map.extend(explicit_substitution.into_iter().map(|r| (r.name, r)));
    } else {
        map.extend(extraction_substitution.into_iter().map(|r| (r.name, r)));
    }
    map
}

pub fn filtered_rules(names: &[&str], use_explicit_subs: bool) -> Vec<Rewrite> {
    let mut all_rules = rules(use_explicit_subs);
    names
        .iter()
        .map(|&n| all_rules.remove(&Symbol::new(n)).expect("rule not found"))
        .collect()
}

// struct BetaApplier {
//     v: Var,
//     e: Var,
//     body: Var,
// }

// impl Applier<RiseLang, RiseAnalysis> for BetaApplier {
//     fn apply_one(
//         &self,
//         egraph: &mut EGraph,
//         eclass: Id,
//         subst: &Subst,
//         _searcher_ast: Option<&PatternAst<RiseLang>>,
//         _rule_name: Symbol,
//     ) -> Vec<Id> {
//         let new_id =
//             substitute::substitute_eclass(egraph, subst[self.v], subst[self.e], subst[self.body]);
//         egraph.union(eclass, new_id);
//         vec![new_id]
//     }
// }

struct BetaExtractApplier {
    v: Var,
    e: Var,
    body: Var,
}

impl Applier<RiseLang, RiseAnalysis> for BetaExtractApplier {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<RiseLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let ex_body = &egraph[subst[self.body]].data.beta_extract;
        let ex_e = &egraph[subst[self.e]].data.beta_extract;
        let v_sym = {
            let ns = &egraph[subst[self.v]].nodes;
            assert!(ns.len() == 1, "expected symbol, found {ns:?}");
            match &ns[0] {
                &RiseLang::Symbol(sym) => sym,
                n => panic!("expected symbol, found {n:?}"),
            }
        };
        let result = substitute::substitute_expr(v_sym, ex_e, ex_body);
        let id = egraph.add_expr(&result);
        egraph.union(eclass, id);
        vec![id]
    }
}

fn with_fresh_var(name: &str, pattern: &str) -> MakeFresh {
    MakeFresh {
        prefix: name[1..].into(),
        fresh: var(name),
        pattern: pattern.parse().unwrap(),
    }
}

struct MakeFresh {
    prefix: String,
    fresh: Var,
    pattern: Pattern<RiseLang>,
}

impl Applier<RiseLang, RiseAnalysis> for MakeFresh {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<RiseLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let sym = RiseLang::Symbol(format!("{}{}", self.prefix, eclass).into());
        let mut new_subst = subst.clone();
        new_subst.insert(self.fresh, egraph.add(sym));
        self.pattern
            .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
    }
}

struct CaptureAvoid {
    fresh: Var,
    v2: Var,
    e: Var,
    if_not_free: Pattern<RiseLang>,
    if_free: Pattern<RiseLang>,
}

impl Applier<RiseLang, RiseAnalysis> for CaptureAvoid {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<RiseLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let e = subst[self.e];
        let v2 = subst[self.v2];
        let v2_free_in_e = egraph[e].data.free.contains(&v2);
        if v2_free_in_e {
            let mut new_subst = subst.clone();
            let sym = RiseLang::Symbol(format!("_{eclass}").into());
            new_subst.insert(self.fresh, egraph.add(sym));
            self.if_free
                .apply_one(egraph, eclass, &new_subst, searcher_ast, rule_name)
        } else {
            self.if_not_free
                .apply_one(egraph, eclass, subst, searcher_ast, rule_name)
        }
    }
}
