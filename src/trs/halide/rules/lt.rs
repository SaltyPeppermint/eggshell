use egg::rewrite as rw;

use crate::trs::halide;
use crate::trs::halide::Rewrite;

pub(crate) fn lt() -> Vec<Rewrite> {
    vec![
        // LT RULES
        rw!("gt-to-lt"      ;  "(> ?x ?z)"              => "(< ?z ?x)"),
        rw!("lt-swap"      ;  "(< ?x ?y)"              => "(< (* -1 ?y) (* -1 ?x))"),
        rw!("lt-to-zero"    ;  "(< ?a ?a)"              => "false"),
        rw!("lt-swap-in"    ;  "(< (+ ?x ?y) ?z)"       => "(< ?x (- ?z ?y))" ),
        rw!("lt-swap-out"   ;  "(< ?z (+ ?x ?y))"       => "(< (- ?z ?y) ?x)" ),
        rw!("lt-x-x-sub-a"  ;  "(< (- ?a ?y) ?a )"      => "true" if halide::is_const_pos("?y")),
        rw!("lt-const-pos"  ;  "(< 0 ?y )"              => "true" if halide::is_const_pos("?y")),
        rw!("lt-const-neg"  ;  "(< ?y 0 )"              => "true" if halide::is_const_neg("?y")),
        rw!("min-lt-cancel" ;  "( < ( min ?x ?y ) ?x )" => "( < ?y ?x )"),
        rw!("lt-min-mutual-term"    ; "( < ( min ?z ?y ) ( min ?x ?y ) )"           => "( < ?z ( min ?x ?y ) )"),
        rw!("lt-max-mutual-term"    ; "( < ( max ?z ?y ) ( max ?x ?y ) )"           => "( < ( max ?z ?y ) ?x )"),
        rw!("lt-min-term-term+pos"  ; "( < ( min ?z ?y ) ( min ?x ( + ?y ?c0 ) ) )" => "( < ( min ?z ?y ) ?x )" if halide::is_const_pos("?c0")),
        rw!("lt-max-term-term+pos"  ; "( < ( max ?z ( + ?y ?c0 ) ) ( max ?x ?y ) )" => "( < ( max ?z ( + ?y ?c0 ) ) ?x )" if halide::is_const_pos("?c0")),
        rw!("lt-min-term+neg-term"  ; "( < ( min ?z ( + ?y ?c0 ) ) ( min ?x ?y ) )" => "( < ( min ?z ( + ?y ?c0 ) ) ?x )" if halide::is_const_neg("?c0")),
        rw!("lt-max-term+neg-term"  ; "( < ( max ?z ?y ) ( max ?x ( + ?y ?c0 ) ) )" => "( < ( max ?z ?y ) ?x )" if halide::is_const_neg("?c0")),
        rw!("lt-min-term+cpos"      ; "( < ( min ?x ?y ) (+ ?x ?c0) )"              => "true" if halide::is_const_pos("?c0")),
        rw!("lt-min-max-cancel"     ; "(< (max ?a ?c) (min ?a ?b))"                 => "false"),
        // rw!("lt-mul-pos-cancel"     ; "(< (* ?x ?y) ?z)"                            => "(< ?x (/ ?z ?y))"  if crate::trs::is_const_pos("?y")),
        // rw!("lt-mul-div-cancel"     ; "(< ?x (/ ?z ?y))"                            => "(< (* ?x ?y) ?z))"  if crate::trs::is_const_pos("?y")),
        //// IS THIS THE CULPRIT?
        rw!("lt-mul-pos-cancel"     ; "(< (* ?x ?y) ?z)"                            => "(< ?x ( / (- ( + ?z ?y ) 1 ) ?y ) ))"  if halide::is_const_pos("?y")),
        rw!("lt-mul-div-cancel"     ; "(< ?y (/ ?x ?z))"                            => "( < ( - ( * ( + ?y 1 ) ?z ) 1 ) ?x )"  if halide::is_const_pos("?z")),
        rw!("lt-const-mod"     ; "(< ?a (% ?x ?b))" => "true"  if halide::compare_constants("?a", "?b", "<=-a")),
        rw!("lt-const-mod-false"     ; "(< ?a (% ?x ?b))" => "false"  if halide::compare_constants("?a", "?b", ">=a")),
    ]
}
