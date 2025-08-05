#[cfg(test)]
mod tests {
    use egg::AstSize;

    use egg::RecExpr;
    use egg::SimpleScheduler;
    use eggshell::eqsat::Eqsat;
    use eggshell::rewrite_system::Halide;
    use eggshell::rewrite_system::RewriteSystem;
    use eggshell::rewrite_system::halide::HalideLang;

    #[test]
    fn simple_eqsat_solved_true() {
        let true_expr:RecExpr<HalideLang> =
       "( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )"
            .parse()
            .unwrap();
        let rules = Halide::full_rules();
        let eqsat = Eqsat::new((&true_expr).into(), &rules);
        let result = eqsat.run(SimpleScheduler);
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as RewriteSystem>::Language::Bool(true),
            expr[0.into()]
        );
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_expr: RecExpr<HalideLang> = "( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap();
        let rules = Halide::full_rules();
        let eqsat = Eqsat::new((&false_expr).into(), &rules);
        let result = eqsat.run(SimpleScheduler);
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as RewriteSystem>::Language::Bool(false),
            expr[0.into()]
        );
    }
}
