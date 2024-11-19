#[cfg(test)]
mod tests {
    use egg::AstSize;

    use eggshell::eqsat::Eqsat;
    use eggshell::trs::Halide;
    use eggshell::trs::TermRewriteSystem;

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt =
        vec!["( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )"
            .parse()
            .unwrap()];
        let rules = Halide::full_rules();
        let eqsat = Eqsat::<Halide>::new(true_stmt);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as TermRewriteSystem>::Language::Bool(true),
            term[0.into()]
        );
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt = vec!["( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap()];
        let rules = Halide::full_rules();
        let eqsat = Eqsat::<Halide>::new(false_stmt);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize);
        assert_eq!(
            <Halide as TermRewriteSystem>::Language::Bool(false),
            term[0.into()]
        );
    }
}
