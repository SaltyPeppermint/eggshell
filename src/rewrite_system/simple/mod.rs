use egg::{Id, Symbol, define_language, rewrite};
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};

use super::{LangExtras, RewriteSystem, SymbolInfo, SymbolType};

pub type Rewrite = egg::Rewrite<SimpleLang, ()>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize, Deserialize, EnumDiscriminants, EnumCount)]
    #[strum_discriminants(derive(EnumIter))]
        pub enum SimpleLang {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Num(i32),
        Symbol(Symbol),
    }
}

impl LangExtras for SimpleLang {
    fn symbol_info(&self) -> SymbolInfo {
        let id = SimpleLangDiscriminants::iter()
            .position(|x| x == self.into())
            .unwrap();
        match self {
            SimpleLang::Symbol(name) => SymbolInfo::new(id, SymbolType::Variable(name.to_string())),
            SimpleLang::Num(value) => SymbolInfo::new(id, SymbolType::Constant(value.to_string())),
            _ => SymbolInfo::new(id, SymbolType::Operator),
        }
    }

    fn operators() -> Vec<&'static str> {
        vec!["+", "*"]
    }

    const MAX_ARITY: usize = 2;
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

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Simple;

impl RewriteSystem for Simple {
    type Language = SimpleLang;
    type Analysis = ();

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        make_rules()
    }
}
