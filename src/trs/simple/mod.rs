use std::cmp::Ordering;
use std::fmt::Display;

use egg::{define_language, rewrite, Id, Symbol};
use serde::Serialize;

use super::{Ruleset, SymbolIter, Trs, TrsError};
use crate::typing::{Type, Typeable, TypingInfo};

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

impl SymbolIter for SimpleLang {
    fn raw_symbols() -> &'static [(&'static str, usize)] {
        &[("+", 2), ("*", 2)]
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

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SimpleRulesets {
    Full,
}

impl Ruleset for SimpleRulesets {
    type Language = SimpleLang;
    type Analysis = ();
    /// takes an class of rules to use then returns the vector of their associated Rewrites
    #[must_use]
    fn rules(&self) -> Vec<Rewrite> {
        make_rules()
    }
}

impl TryFrom<String> for SimpleRulesets {
    type Error = TrsError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "full" | "Full" | "FULL" => Ok(Self::Full),
            _ => Err(Self::Error::BadRulesetName(value)),
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize)]
pub struct Simple;

/// Halide Trs implementation
impl Trs for Simple {
    type Language = SimpleLang;
    type Analysis = ();
    type Rules = SimpleRulesets;
}
