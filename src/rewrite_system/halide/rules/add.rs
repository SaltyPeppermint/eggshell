use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn add() -> Vec<Rewrite> {
    vec![
        // ADD RULES
        rw!("add-comm"      ; "(+ ?a ?b)"                   => "(+ ?b ?a)"),
        rw!("add-assoc"     ; "(+ ?a (+ ?b ?c))"            => "(+ (+ ?a ?b) ?c)"),
        rw!("add-zero"      ; "(+ ?a 0)"                    => "?a"),
        rw!("add-dist-mul"  ; "(* ?a (+ ?b ?c))"            => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("add-fact-mul"  ; "(+ (* ?a ?b) (* ?a ?c))"     => "(* ?a (+ ?b ?c))"),
        rw!("add-denom-mul" ; "(+ (/ ?a ?b) ?c)"            => "(/ (+ ?a (* ?b ?c)) ?b)"),
        rw!("add-denom-div" ; "(/ (+ ?a (* ?b ?c)) ?b)"     => "(+ (/ ?a ?b) ?c)"),
        rw!("add-div-mod"   ; "( + ( / ?x 2 ) ( % ?x 2 ) )" => "( / ( + ?x 1 ) 2 )"),
        // FOLD
        rw!("add-const"     ; "( + (* ?x ?a) (* ?y ?b))"    => "( * (+ (* ?x (/ ?a ?b)) ?y) ?b)" if super::compare_constants("?a", "?b", "%0")),
        // rw!("sub-const-denom-1"; "( - ( / ( + ?x ?y ) ?a ) ( / ( + ?x ?b ) ?a ) )" => "( / ( + ( % ( + ?x ( % ?b ?a ) ) ?a ) ( - ?y ?b ) ) ?a )" if crate::rewrite_system::is_not_zero("?a")),
        // rw!("sub-const-denom-2"; "( - ( / ( + ?x ?c1 ) ?c0 ) ( / ( + ?x ?y ) ?c0 ) )" => "( / ( - ( - (- ( + ?c0 ?c1 ) 1 ) ?y ) ( % ( + ?x ( % ?c1 ?c0 ) ) ?c0 ) ) ?c0 )" if crate::rewrite_system::is_const_pos("?c0")),
    ]
}
