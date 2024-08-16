#[cfg(test)]
mod tests {
    use eggshell::eqsat::Eqsat;
    use eggshell::trs::halide::Halide;
    use eggshell::trs::Trs;
    use eggshell::utils::AstSize2;

    #[test]
    fn simple_eqsat_solved_true() {
        let true_stmt =
        vec!["( == ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) ( + ( * v0 256 ) ( + ( * v1 504 ) v2 ) ) )"
            .parse()
            .unwrap()];
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide>::new(true_stmt);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!("1", term.to_string());
    }

    #[test]
    fn simple_eqsat_solved_false() {
        let false_stmt = vec!["( <= ( + 0 ( / ( + ( % v0 8 ) 167 ) 56 ) ) 0 )"
            .parse()
            .unwrap()];
        let rules = Halide::rules(&Halide::maximum_ruleset());

        let eqsat = Eqsat::<Halide>::new(false_stmt);
        let result = eqsat.run(&rules);
        let root = result.roots().first().unwrap();
        let (_, term) = result.classic_extract(*root, AstSize2);
        assert_eq!("0", term.to_string());
    }
}
