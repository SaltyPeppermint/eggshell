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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pat_rule_1() {
        pat(
            "(typeOf (lam (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt0) ?dt3)) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3)))) (typeOf ?e0 (app (app fun ?dt0) ?dt3))) (app (app fun (app (app arrT ?n1) ?dt0)) (app (app arrT ?n1) ?dt3))) (typeOf (app (typeOf (app (typeOf map (app (app fun (app (app fun ?dt4) ?dt0)) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0)))) (typeOf (lam (typeOf ?e2 ?dt5)) (app (app fun ?dt4) ?dt0))) (app (app fun (app (app arrT ?n1) ?dt4)) (app (app arrT ?n1) ?dt0))) (typeOf %0 (app (app arrT ?n1) ?dt4))) (app (app arrT ?n1) ?dt0))) (app (app arrT ?n1) ?dt3))) (app (app fun (app (app arrT ?n0) ?dt1)) (app (app arrT ?n0) ?dt2)))",
        );
    }
}
