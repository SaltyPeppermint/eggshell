use std::fmt::Display;

use egg::{Analysis, FromOp, Language, Rewrite};
use serde::Serialize;

pub mod arithmatic;
pub mod halide;
pub mod simple;

pub use arithmatic::Arithmatic;
pub use halide::Halide;
pub use simple::Simple;

/// Trait that must be implemented by all Trs consumable by the system
/// It is really simple and breaks down to having a [`Language`] for your System,
/// a [`Analysis`] (can be a simplie as `()`) and one or more `Rulesets` to choose from.
/// The [`Trs::rules`] returns the vector of [`Rewrite`] of your [`Trs`], specified
/// by your ruleset class.
pub trait Trs: Serialize {
    type Language: Language + Display + Sync + Serialize + FromOp + 'static + Sync;
    type Analysis: Analysis<Self::Language, Data: Sync + Serialize + Clone>
        + Clone
        + Sync
        + Serialize
        + Default
        + Sync;

    type Rulesets;

    fn rules(ruleset_class: &Self::Rulesets) -> Vec<Rewrite<Self::Language, Self::Analysis>>;
    fn maximum_ruleset() -> Self::Rulesets;
    // fn prove_goals() -> Vec<Pattern<Self::Language>>;
}

// /// [`EGraph`] parameterized by the Trs
// pub(crate) type TrsEGraph<R> = EGraph<<R as Trs>::Language, <R as Trs>::Analysis>;
