use egg::{Id, Subst, Var};

use super::{data::HalideData, EGraph};

pub mod add;
pub mod and;
pub mod andor;
pub mod div;
pub mod eq;
pub mod ineq;
pub mod lt;
pub mod max;
pub mod min;
pub mod modulo;
pub mod mul;
pub mod not;
pub mod or;
pub mod sub;

/// Checks if a constant is positive
pub fn is_const_pos(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get the constant
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
        // Check if any of the representations of ths constant (nodes inside its eclass) is positive
        // egraph[subst[var]].data.iter().any(|n| match n {
        //     HalideMath::Constant(c) => c > &0,
        //     _ => false,
        // })
        // NEW CORRECT

        match egraph[subst[var]].data {
            Some(HalideData::Int(x)) => x > 0,
            _ => false,
        }
    }
}

/// Checks if a constant is negative
pub fn is_const_neg(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();

    // Get the substitutions where the constant appears
    move |egraph, _, subst| {
        // Check if any of the representations of ths constant (nodes inside its eclass) is negative
        // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
        // egraph[subst[var]].nodes.iter().any(|n| match n {
        //     HalideMath::Constant(c) => c < &0,
        //     _ => false,
        // })
        // NEW CORRECT

        match egraph[subst[var]].data {
            Some(HalideData::Int(x)) => x < 0,
            _ => false,
        }
    }
}

/// Checks if a constant is equals zero
pub fn is_not_zero(var_str: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    // // ACTUALLY FALSE! SEE https://github.com/egraphs-good/egg/issues/297
    // let zero = HalideMath::Constant(0);
    // // Check if any of the representations of the constant (nodes inside its eclass) is zero
    // move |egraph, _, subst| !egraph[subst[var]].nodes.contains(&zero)
    // NEW CORRECT
    move |egraph, _, subst| match egraph[subst[var]].data {
        Some(HalideData::Int(x)) => x != 0,
        _ => false,
    }
}

/// Compares two constants c0 and c1
pub fn compare_constants(
    // first constant
    var_str_1: &str,
    // 2nd constant
    var_str_2: &str,
    // the comparison we're checking
    comp: &'static str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    // Get constants
    let var_1: Var = var_str_1.parse().unwrap();
    let var_2: Var = var_str_2.parse().unwrap();

    move |egraph, _, subst| {
        // Get the eclass of the first constant then match the values of its enodes to check if one of them proves the coming conditions

        let data_1 = egraph[subst[var_1]].data;
        let data_2 = egraph[subst[var_2]].data;

        match (data_1, data_2) {
            (Some(HalideData::Int(c_1)), Some(HalideData::Int(c_2))) => match comp {
                "<" => c_1 < c_2,
                "<a" => c_1 < c_2.abs(),
                "<=" => c_1 <= c_2,
                "<=+1" => c_1 <= (c_2 + 1),
                "<=a" => c_1 <= c_2.abs(),
                "<=-a" => c_1 <= (-c_2.abs()),
                "<=-a+1" => c_1 <= (1 - c_2.abs()),
                ">" => c_1 > c_2,
                ">a" => c_1 > c_2.abs(),
                ">=" => c_1 >= c_2,
                ">=a" => c_1 >= (c_2.abs()),
                ">=a-1" => c_1 >= (c_2.abs() - 1),
                "!=" => c_1 != c_2,
                "%0" => (c_2 != 0) && (c_1 % c_2 == 0),
                "!%0" => (c_2 != 0) && (c_1 % c_2 != 0),
                "%0<" => (c_2 > 0) && (c_1 % c_2 == 0),
                "%0>" => (c_2 < 0) && (c_1 % c_2 == 0),
                _ => false,
            },
            _ => false,
        }
    }
}
