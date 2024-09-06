use std::fmt::Display;

use egg::{define_language, rewrite, Id, RecExpr, Symbol};
use serde::Serialize;

use crate::typing::{Typeable, TypingError};

use super::{Trs, TrsError};

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

impl TryFrom<String> for Ruleset {
    type Error = TrsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            _ => Err(Self::Error::BadRulesetName(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SimpleType {
    #[default]
    Integer,
}

impl Display for SimpleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer => write!(f, "Integer"),
        }
    }
}

impl Typeable for SimpleLang {
    type Type = SimpleType;

    fn type_node(&self, _: &RecExpr<Self>) -> Result<Self::Type, TypingError> {
        Ok(Self::Type::Integer)
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Simple;

/// Halide Trs implementation
impl Trs for Simple {
    type Language = SimpleLang;
    type Analysis = ();
    type Rulesets = Ruleset;

    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[must_use]
    fn rules(_ruleset_class: &Ruleset) -> Vec<Rewrite> {
        make_rules()
    }
}
