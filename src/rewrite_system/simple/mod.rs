use egg::{Id, Symbol, rewrite};
use serde::{Deserialize, Serialize};

type Rewrite = egg::Rewrite<SimpleLang, ()>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

egg::define_language! {
    #[derive(Serialize, Deserialize)]
        pub enum SimpleLang {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Num(i32),
        Symbol(Symbol),
    }
}

#[must_use]
pub fn rules() -> Vec<Rewrite> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}
