use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn ineq() -> Vec<Rewrite> {
    vec![
        // Inequality RULES
        rw!("ineq-to-eq";  "(!= ?x ?y)"        => "(! (== ?x ?y))"),
    ]
}
