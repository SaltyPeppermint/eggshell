use std::cmp::Ordering;
use std::fmt::Display;

use egg::{define_language, rewrite, Id, Symbol};
use serde::Serialize;

use super::TermRewriteSystem;
use crate::{
    features::{AsFeatures, Featurizer, SymbolType},
    typing::{Type, Typeable, TypingInfo},
};

pub type Rewrite = egg::Rewrite<SimpleLang, ()>;

// Big thanks to egg, this is mostly copy-pasted from their tests folder

define_language! {
    #[derive(Serialize)]
    pub enum SimpleLang {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

impl AsFeatures for SimpleLang {
    fn symbol_list(variable_names: Vec<String>) -> Featurizer<Self> {
        Featurizer::new(
            vec![
                SimpleLang::Num(0),
                SimpleLang::Add([0.into(), 0.into()]),
                SimpleLang::Mul([0.into(), 0.into()]),
                // We do not include symbols!
                // SimpleLang::Symbol(Symbol::new("")),
            ],
            variable_names,
        )
    }

    fn symbol_type(&self) -> SymbolType {
        match self {
            SimpleLang::Symbol(name) => SymbolType::Variable(name.as_str()),
            SimpleLang::Num(value) => SymbolType::Constant((*value).into()),
            _ => SymbolType::Operator,
        }
    }

    fn into_symbol(name: String) -> Self {
        SimpleLang::Symbol(name.into())
    }
}

impl Typeable for SimpleLang {
    type Type = SimpleType;

    fn type_info(&self) -> TypingInfo<Self::Type> {
        TypingInfo::new(Self::Type::Top, Self::Type::Top)
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
pub enum SimpleType {
    Top,
    Bottom,
}

impl Display for SimpleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Top => write!(f, "Top (Integer)"),
            Self::Bottom => write!(f, "Bottom"),
        }
    }
}

impl PartialOrd for SimpleType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            // Bottom type is less than everything else
            (Self::Top, Self::Bottom) => Some(Ordering::Greater),
            (Self::Bottom, Self::Top) => Some(Ordering::Less),
            (Self::Bottom, Self::Bottom) | (Self::Top, Self::Top) => Some(Ordering::Equal),
        }
    }
}

impl Type for SimpleType {
    fn top() -> Self {
        Self::Top
    }

    fn bottom() -> Self {
        Self::Bottom
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

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Simple;

impl TermRewriteSystem for Simple {
    type Language = SimpleLang;
    type Analysis = ();

    fn full_rules() -> Vec<egg::Rewrite<Self::Language, Self::Analysis>> {
        make_rules()
    }
}
