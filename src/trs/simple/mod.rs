use egg::{define_language, rewrite, Id, Symbol};
use serde::Serialize;

use super::Trs;

pub type Rewrite = egg::Rewrite<SimpleLanguage, ()>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize)]
    pub enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<Rewrite> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Ruleset {
    Full,
}

/// Halide Trs implementation
#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Simple;

impl Trs for Simple {
    type Language = SimpleLanguage;
    type Analysis = ();
    type Rulesets = Ruleset;

    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[allow(clippy::similar_names)]
    #[must_use]
    fn rules(_ruleset_class: &Ruleset) -> Vec<Rewrite> {
        make_rules()
    }

    #[must_use]
    fn maximum_ruleset() -> Self::Rulesets {
        Ruleset::Full
    }

    // #[must_use]
    // fn prove_goals() -> Vec<egg::Pattern<Self::Language>> {
    //     panic!("THERE ARE NO PROVE GOALS HERE")
    // }
}
