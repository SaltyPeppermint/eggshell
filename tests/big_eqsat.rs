#[cfg(test)]
mod tests {
    use egg::AstSize;

    use eggshell::eqsat::Eqsat;
    use eggshell::eqsat::StartMaterial;
    use eggshell::trs::Halide;
    use eggshell::trs::TermRewriteSystem;

    #[test]
    fn simple_eqsat_solved_true() {
        let true_expr =
       [&"( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )"
            .parse()
            .unwrap()];
        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&true_expr), &rules);
        let result = eqsat.run();
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as TermRewriteSystem>::Language::Bool(true),
            expr[0.into()]
        );
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_expr = [&"( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap()];
        let rules = Halide::full_rules();
        let eqsat = Eqsat::new(StartMaterial::RecExprs(&false_expr), &rules);
        let result = eqsat.run();
        let root = result.roots().first().unwrap();
        let (_, expr) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as TermRewriteSystem>::Language::Bool(false),
            expr[0.into()]
        );
    }
}
