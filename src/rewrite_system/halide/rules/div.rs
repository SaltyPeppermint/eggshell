use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn div() -> Vec<Rewrite> {
    vec![
        // DIV RULES
        rw!("div-zero"      ; "(/ 0 ?a)"            => "0" if super::is_not_zero("?a")),
        rw!("div-cancel"    ; "(/ ?a ?a)"           => "1" if super::is_not_zero("?a")),
        rw!("div-minus-down"; "(/ (* -1 ?a) ?b)"    => "(/ ?a (* -1 ?b))"),
        rw!("div-minus-up"  ; "(/ ?a (* -1 ?b))"    => "(/ (* -1 ?a) ?b)"),
        rw!("div-minus-in"  ; "(* -1 (/ ?a ?b))"    => "(/ (* -1 ?a) ?b)"),
        rw!("div-minus-out" ; "(/ (* -1 ?a) ?b)"    => "(* -1 (/ ?a ?b))"),
        // FOLD
        rw!("div-consts-div"; "( / ( * ?x ?a ) ?b )" => "( / ?x ( / ?b ?a ) )" if super::compare_constants("?b", "?a", "%0<")),
        rw!("div-consts-mul"; "( / ( * ?x ?a ) ?b )" => "( * ?x ( / ?a ?b ) )" if super::compare_constants("?a", "?b", "%0<")),
        rw!("div-consts-add"; "( / ( + ( * ?x ?a ) ?y ) ?b )" => "( + ( * ?x ( / ?a ?b ) ) ( / ?y ?b ) )" if super::compare_constants("?a", "?b", "%0<")),
        rw!("div-separate"  ; "( / ( + ?x ?a ) ?b )" => "( + ( / ?x ?b ) ( / ?a ?b ) )" if super::compare_constants("?a", "?b", "%0<")),
    ]
}
