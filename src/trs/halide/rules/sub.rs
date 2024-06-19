use egg::rewrite as rw;

use crate::trs::halide::Rewrite;

pub(crate) fn sub() -> Vec<Rewrite> {
    vec![
        // SUB RULES
        rw!("sub-to-add"; "(- ?a ?b)"   => "(+ ?a (* -1 ?b))"),
        // rw!("add-to-sub"; "(+ ?a ?b)"   => "(- ?a (* -1 ?b))"),
    ]
}
