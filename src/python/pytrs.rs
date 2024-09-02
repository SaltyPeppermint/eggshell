use super::macros::monomorphize;

pub mod arithmatic {
    super::monomorphize!(crate::trs::Arithmetic);
}

pub mod halide {
    super::monomorphize!(crate::trs::Halide);
}

pub mod simple {
    super::monomorphize!(crate::trs::Simple);
}
