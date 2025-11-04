use egg::{Applier, Pattern};

use super::{Rise, RiseAnalysis};

pub fn pat(pat: &str) -> impl Applier<Rise, RiseAnalysis> {
    pat.parse::<Pattern<Rise>>().unwrap()
}

pub fn not_free_in<A>(var: &str, index: u32, applier: A) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}

pub fn shifted<A>(
    var: &str,
    shifted_var: &str,
    shift: i32,
    cutoff: u32,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}

pub fn shifted_check<A>(
    var: &str,
    shifted_var: &str,
    shift: i32,
    cutoff: u32,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}

pub fn compute_nat<A>(var: &str, nat_pattern: &str, applier: A) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}

pub fn compute_nat_check<A>(
    var: &str,
    nat_pattern: &str,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}

pub fn vectorize_scalar_fun<A>(
    var: &str,
    size_var: &str,
    vectorized_var: &str,
    applier: A,
) -> impl Applier<Rise, RiseAnalysis>
where
    A: Applier<Rise, RiseAnalysis>,
{
    pat(")unimplemented(")
}
