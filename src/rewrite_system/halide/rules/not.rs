use egg::rewrite as rw;

use crate::rewrite_system::halide::Rewrite;

pub(crate) fn not() -> Vec<Rewrite> {
    vec![
        // NOT RULES
        rw!("eqlt-to-not-gt";  "(<= ?x ?y)"     => "(! (< ?y ?x))" ),
        rw!("not-gt-to-eqlt";  "(! (< ?y ?x))"  => "(<= ?x ?y)" ),
        rw!("eqgt-to-not-lt";  "(>= ?x ?y)"     => "(! (< ?x ?y))" ),
        rw!("not-eq-to-ineq";  "(! (== ?x ?y))" => "(!= ?x ?y)" ),
        rw!("not-not"       ;  "(! (! ?x))"     => "?x" ),
    ]
}
