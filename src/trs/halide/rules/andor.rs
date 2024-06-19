use egg::rewrite as rw;

use crate::trs::halide::Rewrite;

pub(crate) fn andor() -> Vec<Rewrite> {
    vec![
        // AND-OR RULES
        rw!("and-over-or"   ;  "(&& ?a (|| ?b ?c))"        => "(|| (&& ?a ?b) (&& ?a ?c))"),
        rw!("or-over-and"   ;  "(|| ?a (&& ?b ?c))"        => "(&& (|| ?a ?b) (|| ?a ?c))"),
        rw!("or-x-and-x-y"  ;  "(|| ?x (&& ?x ?y))"        => "?x"),
    ]
}
