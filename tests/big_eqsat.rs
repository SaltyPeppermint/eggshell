#[cfg(test)]
mod tests {
    use egg::AstSize;

    use egg::Extractor;
    use egg::RecExpr;
    use egg::SimpleScheduler;
    use eggshell::eqsat;
    use eggshell::eqsat::EqsatConf;
    use eggshell::rewrite_system::halide;
    use eggshell::rewrite_system::halide::HalideLang;

    #[test]
    fn simple_eqsat_solved_true() {
        let true_expr:RecExpr<HalideLang> =
       "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )"
            .parse()
            .unwrap();
        let rules = halide::rules(halide::HalideRuleset::Full);

        let (runner, roots) = eqsat::eqsat(
            &EqsatConf::default(),
            (&true_expr).into(),
            &rules,
            None,
            SimpleScheduler,
        );
        let root = roots.first().unwrap();
        let (_, expr) = Extractor::new(&runner.egraph, AstSize).find_best(*root);
        assert_eq!(HalideLang::Bool(true), expr[0.into()]);
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_expr: RecExpr<HalideLang> = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap();
        let rules = halide::rules(halide::HalideRuleset::Full);
        let (runner, roots) = eqsat::eqsat(
            &EqsatConf::default(),
            (&false_expr).into(),
            &rules,
            None,
            SimpleScheduler,
        );
        let root = roots.first().unwrap();
        let (_, expr) = Extractor::new(&runner.egraph, AstSize).find_best(*root);
        assert_eq!(HalideLang::Bool(false), expr[0.into()]);
    }
}
