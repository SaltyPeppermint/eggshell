use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn or() -> Vec<Rewrite> {
    vec![
        // OR RULES
        rw!("or-to-and" ;"(|| ?x ?y)"        => "(! (&& (! ?x) (! ?y)))"),
        rw!("or-comm"   ;"(|| ?y ?x)"        => "(|| ?x ?y)"),
    ]
}
