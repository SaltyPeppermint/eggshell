use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn eq() -> Vec<Rewrite> {
    vec![
        // Equality RULES
        rw!("eq-comm"       ; "(== ?x ?y)"           => "(== ?y ?x)"),
        rw!("eq-x-y-0"      ; "(== ?x ?y)"           => "(== (- ?x ?y) 0)"),
        rw!("eq-swap"       ; "(== (+ ?x ?y) ?z)"    => "(== ?x (- ?z ?y))"),
        rw!("eq-x-x"        ; "(== ?x ?x)"           => "true"),
        rw!("eq-mul-x-y-0"  ; "(== (* ?x ?y) 0)"     => "(|| (== ?x 0) (== ?y 0))"),
        rw!("eq-max-lt"     ; "( == (max ?x ?y) ?y)" => "(<= ?x ?y)"),
        rw!("Eq-min-lt"     ; "( == (min ?x ?y) ?y)" => "(<= ?y ?x)"),
        rw!("Eq-lt-min"     ; "(<= ?y ?x)"           => "( == (min ?x ?y) ?y)"),
        rw!("Eq-a-b"        ; "(== (* ?a ?x) ?b)"    => "false" if super::compare_constants("?b", "?a", "!%0")),
        rw!("Eq-max-c-pos"  ; "(== (max ?x ?c) 0)"   => "false" if super::is_const_pos("?c")),
        rw!("Eq-max-c-neg"  ; "(== (max ?x ?c) 0)"   => "(== ?x 0)" if super::is_const_neg("?c")),
        rw!("Eq-min-c-pos"  ; "(== (min ?x ?c) 0)"   => "false" if super::is_const_neg("?c")),
        rw!("Eq-min-c-neg"  ; "(== (min ?x ?c) 0)"   => "(== ?x 0)" if super::is_const_pos("?c")),
    ]
}
