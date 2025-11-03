use super::{ConstantFold, Math};
use egg::{Rewrite, rewrite as rw};

#[expect(clippy::too_many_lines)]
pub fn rules() -> Vec<Rewrite<Math, ConstantFold>> {
    vec![
        // Commutativity
        rw!("add_commutative"; "(+ ?a ?b)" => "(+ ?b ?a)"), // +-commutative
        rw!("mul_commutative"; "(* ?a ?b)" => "(* ?b ?a)"), // *-commutative
        // Associativity
        rw!("associate_add_r_add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"), // associate-+r+
        rw!("associate_add_l_add"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"), // associate-+l+
        rw!("associate_add_r_sub"; "(+ ?a (- ?b ?c))" => "(- (+ ?a ?b) ?c)"), // associate-+r-
        rw!("associate_add_l_sub"; "(+ (- ?a ?b) ?c)" => "(- ?a (- ?b ?c))"), // associate-+l-
        rw!("associate_sub_r_add"; "(- ?a (+ ?b ?c))" => "(- (- ?a ?b) ?c)"), // associate--r+
        rw!("associate_sub_l_add"; "(- (+ ?a ?b) ?c)" => "(+ ?a (- ?b ?c))"), // associate--l+
        rw!("associate_sub_l_sub"; "(- (- ?a ?b) ?c)" => "(- ?a (+ ?b ?c))"), // associate--l-
        rw!("associate_sub_r_sub"; "(- ?a (- ?b ?c))" => "(+ (- ?a ?b) ?c)"), // associate--r-
        rw!("associate_mul_r_mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"), // associate-*r*
        rw!("associate_mul_l_mul"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"), // associate-*l*
        rw!("associate_mul_r_div"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)"), // associate-*r/
        rw!("associate_mul_l_div"; "(* (/ ?a ?b) ?c)" => "(/ (* ?a ?c) ?b)"), // associate-*l/
        rw!("associate_div_r_mul"; "(/ ?a (* ?b ?c))" => "(/ (/ ?a ?b) ?c)"), // associate-/r*
        rw!("associate_div_r_div"; "(/ ?a (/ ?b ?c))" => "(* (/ ?a ?b) ?c)"), // associate-/r/
        rw!("associate_div_l_div"; "(/ (/ ?b ?c) ?a)" => "(/ ?b (* ?c ?a))"), // associate-/l/
        rw!("associate_div_l_mul"; "(/ (* ?b ?c) ?a)" => "(* ?b (/ ?c ?a))"), // associate-/l*
        // Identity
        rw!("remove_double_div"; "(/ 1 (/ 1 ?a))" => "?a"), // remove-double-div
        rw!("rgt_mul_inverse"; "(* ?a (/ 1 ?a))" => "1"),   // rgt-mult-inverse
        rw!("lft_mul_inverse"; "(* (/ 1 ?a) ?a)" => "1"),   // lft-mult-inverse
        rw!("add_inverses"; "(- ?a ?a)" => "0"),            // +-inverses
        rw!("div0"; "(/ 0 ?a)" => "0"),                     // div0
        rw!("mul0_lft"; "(* 0 ?a)" => "0"),                 // mul0-lft
        rw!("mul0_rgt"; "(* ?a 0)" => "0"),                 // mul0-rgt
        rw!("mul_inverses"; "(/ ?a ?a)" => "1"),            // *-inverses
        rw!("add_lft_identity"; "(+ 0 ?a)" => "?a"),        // +-lft-identity
        rw!("add_rgt_identity"; "(+ ?a 0)" => "?a"),        // +-rgt-identity
        rw!("sub_rgt_identity"; "(- ?a 0)" => "?a"),        // --rgt-identity
        rw!("sub0_neg"; "(- 0 ?a)" => "(neg ?a)"),          // sub0-neg
        rw!("remove_double_neg"; "(neg (neg ?a))" => "?a"), // remove-double-neg
        rw!("mul_lft_identity"; "(* 1 ?a)" => "?a"),        // *-lft-identity
        rw!("mul_rgt_identity"; "(* ?a 1)" => "?a"),        // *-rgt-identity
        rw!("div_rgt_identity"; "(/ ?a 1)" => "?a"),        // /-rgt-identity
        rw!("mul_1_neg"; "(* -1 ?a)" => "(neg ?a)"),        // mul-1-neg
        // Counting
        rw!("count_2"; "(+ ?x ?x)" => "(* 2 ?x)"), // count-2
        rw!("two_split"; "2" => "(+ 1 1)"),        // 2-split
        rw!("count_2_rev"; "(* 2 ?x)" => "(+ ?x ?x)"), // count-2-rev
        // Distributivity
        rw!("distribute_lft_in"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"), // distribute-lft-in
        rw!("distribute_rgt_in"; "(* ?a (+ ?b ?c))" => "(+ (* ?b ?a) (* ?c ?a))"), // distribute-rgt-in
        rw!("distribute_lft_out"; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"), // distribute-lft-out
        rw!("distribute_lft_out_sub"; "(- (* ?a ?b) (* ?a ?c))" => "(* ?a (- ?b ?c))"), // distribute-lft-out--
        rw!("distribute_rgt_out"; "(+ (* ?b ?a) (* ?c ?a))" => "(* ?a (+ ?b ?c))"), // distribute-rgt-out
        rw!("distribute_rgt_out_sub"; "(- (* ?b ?a) (* ?c ?a))" => "(* ?a (- ?b ?c))"), // distribute-rgt-out--
        rw!("distribute_lft1_in"; "(+ (* ?b ?a) ?a)" => "(* (+ ?b 1) ?a)"), // distribute-lft1-in
        rw!("distribute_rgt1_in"; "(+ ?a (* ?c ?a))" => "(* (+ ?c 1) ?a)"), // distribute-rgt1-in
        // Safe Distributivity
        rw!("distribute_lft_neg_in"; "(neg (* ?a ?b))" => "(* (neg ?a) ?b)"), // distribute-lft-neg-in
        rw!("distribute_rgt_neg_in"; "(neg (* ?a ?b))" => "(* ?a (neg ?b))"), // distribute-rgt-neg-in
        rw!("distribute_lft_neg_out"; "(* (neg ?a) ?b)" => "(neg (* ?a ?b))"), // distribute-lft-neg-out
        rw!("distribute_rgt_neg_out"; "(* ?a (neg ?b))" => "(neg (* ?a ?b))"), // distribute-rgt-neg-out
        rw!("distribute_neg_in"; "(neg (+ ?a ?b))" => "(+ (neg ?a) (neg ?b))"), // distribute-neg-in
        rw!("distribute_neg_out"; "(+ (neg ?a) (neg ?b))" => "(neg (+ ?a ?b))"), // distribute-neg-out
        rw!("distribute_frac_neg"; "(/ (neg ?a) ?b)" => "(neg (/ ?a ?b))"), // distribute-frac-neg
        rw!("distribute_frac_neg2"; "(/ ?a (neg ?b))" => "(neg (/ ?a ?b))"), // distribute-frac-neg2
        rw!("distribute_neg_frac"; "(neg (/ ?a ?b))" => "(/ (neg ?a) ?b)"), // distribute-neg-frac
        rw!("distribute_neg_frac2"; "(neg (/ ?a ?b))" => "(/ ?a (neg ?b))"), // distribute-neg-frac2
        rw!("fp_cancel_sign_sub"; "(- ?a (* (neg ?b) ?c))" => "(+ ?a (* ?b ?c))"), // fp-cancel-sign-sub
        rw!("fp_cancel_sub_sign"; "(+ ?a (* (neg ?b) ?c))" => "(- ?a (* ?b ?c))"), // fp-cancel-sub-sign
        rw!("fp_cancel_sign_sub_inv"; "(+ ?a (* ?b ?c))" => "(- ?a (* (neg ?b) ?c))"), // fp-cancel-sign-sub-inv
        rw!("fp_cancel_sub_sign_inv"; "(- ?a (* ?b ?c))" => "(+ ?a (* (neg ?b) ?c))"), // fp-cancel-sub-sign-inv
        // Sub/neg conversions
        rw!("sub_flip"; "(- ?a ?b)" => "(+ ?a (neg ?b))"), // sub-flip
        rw!("sub_flip_reverse"; "(+ ?a (neg ?b))" => "(- ?a ?b)"), // sub-flip-reverse
        rw!("sub_negate"; "(neg (- ?b ?a))" => "(- ?a ?b)"), // sub-negate
        rw!("sub_negate_rev"; "(- ?a ?b)" => "(neg (- ?b ?a))"), // sub-negate-rev
        rw!("add_flip"; "(+ ?a ?b)" => "(- ?a (neg ?b))"), // add-flip
        rw!("add_flip_rev"; "(- ?a (neg ?b))" => "(+ ?a ?b)"), // add-flip-rev
        // Difference of squares
        rw!("swap_sqr"; "(* (* ?a ?b) (* ?a ?b))" => "(* (* ?a ?a) (* ?b ?b))"), // swap-sqr
        rw!("unswap_sqr"; "(* (* ?a ?a) (* ?b ?b))" => "(* (* ?a ?b) (* ?a ?b))"), // unswap-sqr
        rw!("difference_of_squares"; "(- (* ?a ?a) (* ?b ?b))" => "(* (+ ?a ?b) (- ?a ?b))"), // difference-of-squares
        rw!("difference_of_sqr_1"; "(- (* ?a ?a) 1)" => "(* (+ ?a 1) (- ?a 1))"), // difference-of-sqr-1
        rw!("difference_of_sqr_neg1"; "(+ (* ?a ?a) -1)" => "(* (+ ?a 1) (- ?a 1))"), // difference-of-sqr--1
        rw!("pow_sqr"; "(* (pow ?a ?b) (pow ?a ?b))" => "(pow ?a (* 2 ?b))"),         // pow-sqr
        rw!("sum_square_pow"; "(pow (+ ?a ?b) 2)" => "(+ (+ (pow ?a 2) (* 2 (* ?a ?b))) (pow ?b 2))"), // sum-square-pow
        rw!("sub_square_pow"; "(pow (- ?a ?b) 2)" => "(+ (- (pow ?a 2) (* 2 (* ?a ?b))) (pow ?b 2))"), // sub-square-pow
        rw!("sum_square_pow_rev"; "(+ (+ (pow ?a 2) (* 2 (* ?a ?b))) (pow ?b 2))" => "(pow (+ ?a ?b) 2)"), // sum-square-pow-rev
        rw!("sub_square_pow_rev"; "(+ (- (pow ?a 2) (* 2 (* ?a ?b))) (pow ?b 2))" => "(pow (- ?a ?b) 2)"), // sub-square-pow-rev
        rw!("difference_of_sqr_1_rev"; "(* (+ ?a 1) (- ?a 1))" => "(- (* ?a ?a) 1)"), // difference-of-sqr-1-rev
        rw!("difference_of_sqr_neg1_rev"; "(* (+ ?a 1) (- ?a 1))" => "(+ (* ?a ?a) -1)"), // difference-of-sqr--1-rev
        rw!("difference_of_squares_rev"; "(* (+ ?a ?b) (- ?a ?b))" => "(- (* ?a ?a) (* ?b ?b))"), // difference-of-squares-rev
        // Mul/div flip
        rw!("mult_flip"; "(/ ?a ?b)" => "(* ?a (/ 1 ?b))"), // mult-flip
        rw!("mult_flip_rev"; "(* ?a (/ 1 ?b))" => "(/ ?a ?b)"), // mult-flip-rev
        // rw!("div_flip"; "(/ ?a ?b)" => "(sound-/ 1 (sound-/ ?b ?a 0) (/ ?a ?b))"), // div-flip (requires sound-)
        rw!("div_flip_rev"; "(/ 1 (/ ?b ?a))" => "(/ ?a ?b)"), // div-flip-rev
        // Fractions
        // rw!("sum_to_mult"; "(+ ?a ?b)" => "(* (+ 1 (/ ?b ?a)) ?a)"), // sum-to-mult #:unsound
        rw!("sum_to_mult_rev"; "(* (+ 1 (/ ?b ?a)) ?a)" => "(+ ?a ?b)"), // sum-to-mult-rev
        // rw!("sub_to_mult"; "(- ?a ?b)" => "(* (- 1 (/ ?b ?a)) ?a)"), // sub-to-mult #:unsound
        rw!("sub_to_mult_rev"; "(* (- 1 (/ ?b ?a)) ?a)" => "(- ?a ?b)"), // sub-to-mult-rev
        rw!("add_to_fraction"; "(+ ?c (/ ?b ?a))" => "(/ (+ (* ?c ?a) ?b) ?a)"), // add-to-fraction
        rw!("add_to_fraction_rev"; "(/ (+ (* ?c ?a) ?b) ?a)" => "(+ ?c (/ ?b ?a))"), // add-to-fraction-rev
        rw!("sub_to_fraction"; "(- ?c (/ ?b ?a))" => "(/ (- (* ?c ?a) ?b) ?a)"), // sub-to-fraction
        rw!("sub_to_fraction_rev"; "(/ (- (* ?c ?a) ?b) ?a)" => "(- ?c (/ ?b ?a))"), // sub-to-fraction-rev
        rw!("common_denominator"; "(+ (/ ?a ?b) (/ ?c ?d))" => "(/ (+ (* ?a ?d) (* ?c ?b)) (* ?b ?d))"), // common-denominator
        // Polynomial flip rules (require sound-)
        // rw!("flip_add"; "(+ ?a ?b)" => "(sound-/ (- (* ?a ?a) (* ?b ?b)) (- ?a ?b) (+ ?a ?b))"), // flip-+
        // rw!("flip_sub"; "(- ?a ?b)" => "(sound-/ (- (* ?a ?a) (* ?b ?b)) (+ ?a ?b) (- ?a ?b))"), // flip--

        // Difference of cubes
        rw!("sum_cubes"; "(+ (pow ?a 3) (pow ?b 3))" => "(* (+ (* ?a ?a) (- (* ?b ?b) (* ?a ?b))) (+ ?a ?b))"), // sum-cubes
        rw!("difference_cubes"; "(- (pow ?a 3) (pow ?b 3))" => "(* (+ (* ?a ?a) (+ (* ?b ?b) (* ?a ?b))) (- ?a ?b))"), // difference-cubes
        rw!("difference_cubes_rev"; "(* (+ (* ?a ?a) (+ (* ?b ?b) (* ?a ?b))) (- ?a ?b))" => "(- (pow ?a 3) (pow ?b 3))"), // difference-cubes-rev
        rw!("sum_cubes_rev"; "(* (+ (* ?a ?a) (- (* ?b ?b) (* ?a ?b))) (+ ?a ?b))" => "(+ (pow ?a 3) (pow ?b 3))"), // sum-cubes-rev
        // Polynomial flip3 (require sound-)
        // rw!("flip3_add"; "(+ ?a ?b)" => "(sound-/ (+ (pow ?a 3) (pow ?b 3)) (+ (* ?a ?a) (- (* ?b ?b) (* ?a ?b))) (+ ?a ?b))"), // flip3-+
        // rw!("flip3_sub"; "(- ?a ?b)" => "(sound-/ (- (pow ?a 3) (pow ?b 3)) (+ (* ?a ?a) (+ (* ?b ?b) (* ?a ?b))) (- ?a ?b))"), // flip3--

        // Dealing with fractions
        rw!("div_sub"; "(/ (- ?a ?b) ?c)" => "(- (/ ?a ?c) (/ ?b ?c))"), // div-sub
        rw!("times_frac"; "(/ (* ?a ?b) (* ?c ?d))" => "(* (/ ?a ?c) (/ ?b ?d))"), // times-frac
        rw!("div_add"; "(/ (+ ?a ?b) ?c)" => "(+ (/ ?a ?c) (/ ?b ?c))"), // div-add
        rw!("div_add_rev"; "(+ (/ ?a ?c) (/ ?b ?c))" => "(/ (+ ?a ?b) ?c)"), // div-add-rev
        rw!("sub_div"; "(- (/ ?a ?c) (/ ?b ?c))" => "(/ (- ?a ?b) ?c)"), // sub-div
        rw!("frac_add"; "(+ (/ ?a ?b) (/ ?c ?d))" => "(/ (+ (* ?a ?d) (* ?b ?c)) (* ?b ?d))"), // frac-add
        rw!("frac_sub"; "(- (/ ?a ?b) (/ ?c ?d))" => "(/ (- (* ?a ?d) (* ?b ?c)) (* ?b ?d))"), // frac-sub
        rw!("frac_times"; "(* (/ ?a ?b) (/ ?c ?d))" => "(/ (* ?a ?c) (* ?b ?d))"), // frac-times
        rw!("frac_2neg"; "(/ ?a ?b)" => "(/ (neg ?a) (neg ?b))"),                  // frac-2neg
        rw!("frac_2neg_rev"; "(/ (neg ?a) (neg ?b))" => "(/ ?a ?b)"),              // frac-2neg-rev
        // Square root
        rw!("rem_square_sqrt"; "(* (sqrt ?x) (sqrt ?x))" => "?x"), // rem-square-sqrt
        rw!("rem_sqrt_square"; "(sqrt (* ?x ?x))" => "(fabs ?x)"), // rem-sqrt-square
        rw!("rem_sqrt_square_rev"; "(fabs ?x)" => "(sqrt (* ?x ?x))"), // rem-sqrt-square-rev
        rw!("sqr_neg"; "(* (neg ?x) (neg ?x))" => "(* ?x ?x)"),    // sqr-neg
        rw!("sqr_abs"; "(* (fabs ?x) (fabs ?x))" => "(* ?x ?x)"),  // sqr-abs
        rw!("sqr_abs_rev"; "(* ?x ?x)" => "(* (fabs ?x) (fabs ?x))"), // sqr-abs-rev
        rw!("sqr_neg_rev"; "(* ?x ?x)" => "(* (neg ?x) (neg ?x))"), // sqr-neg-rev
        rw!("sqrt_cbrt"; "(sqrt (cbrt ?x))" => "(cbrt (sqrt ?x))"), // sqrt-cbrt
        rw!("cbrt_sqrt"; "(cbrt (sqrt ?x))" => "(sqrt (cbrt ?x))"), // cbrt-sqrt
        // Absolute value
        rw!("fabs_fabs"; "(fabs (fabs ?x))" => "(fabs ?x)"), // fabs-fabs
        rw!("fabs_sub"; "(fabs (- ?a ?b))" => "(fabs (- ?b ?a))"), // fabs-sub
        rw!("fabs_add"; "(fabs (+ (fabs ?a) (fabs ?b)))" => "(+ (fabs ?a) (fabs ?b))"), // fabs-add
        rw!("fabs_neg"; "(fabs (neg ?x))" => "(fabs ?x)"),   // fabs-neg
        rw!("fabs_sqr"; "(fabs (* ?x ?x))" => "(* ?x ?x)"),  // fabs-sqr
        rw!("fabs_mul"; "(fabs (* ?a ?b))" => "(* (fabs ?a) (fabs ?b))"), // fabs-mul
        rw!("fabs_div"; "(fabs (/ ?a ?b))" => "(/ (fabs ?a) (fabs ?b))"), // fabs-div
        rw!("neg_fabs"; "(fabs ?x)" => "(fabs (neg ?x))"),   // neg-fabs
        rw!("mul_fabs"; "(* (fabs ?a) (fabs ?b))" => "(fabs (* ?a ?b))"), // mul-fabs
        rw!("div_fabs"; "(/ (fabs ?a) (fabs ?b))" => "(fabs (/ ?a ?b))"), // div-fabs
        rw!("sqrt_fabs"; "(fabs (sqrt ?a))" => "(sqrt ?a)"), // sqrt-fabs
        rw!("sqrt_fabs_rev"; "(sqrt ?a)" => "(fabs (sqrt ?a))"), // sqrt-fabs-rev
        // The following two rules reference 'copysign', which is not in the Math enum; commented out.
        // rw!("fabs_lhs_div"; "(/ (fabs ?x) ?x)" => "(copysign 1 ?x)"), // fabs-lhs-div
        // rw!("fabs_rhs_div"; "(/ ?x (fabs ?x))" => "(copysign 1 ?x)"), // fabs-rhs-div
        // rw!("fabs_cbrt"; "(fabs (/ (cbrt ?a) ?a))" => "(/ (cbrt ?a) ?a)"), // fabs-cbrt
        // rw!("fabs_cbrt_rev"; "(/ (cbrt ?a) ?a)" => "(fabs (/ (cbrt ?a) ?a))"), // fabs-cbrt-rev

        // Copysign (not in Math enum, commented out)
        // rw!("copysign_neg"; "(copysign ?a (neg ?b))" => "(neg (copysign ?a ?b))"), // copysign-neg
        // rw!("neg_copysign"; "(neg (copysign ?a ?b))" => "(copysign ?a (neg ?b))"), // neg-copysign
        // rw!("copysign_other_neg"; "(copysign (neg ?a) ?b)" => "(copysign ?a ?b)"), // copysign-other-neg
        // rw!("copysign_fabs"; "(copysign ?a (fabs ?b))" => "(fabs ?a)"), // copysign-fabs
        // rw!("copysign_other_fabs"; "(copysign (fabs ?a) ?b)" => "(copysign ?a ?b)"), // copysign-other-fabs
        // rw!("fabs_copysign"; "(fabs (copysign ?a ?b))" => "(fabs ?a)"), // fabs-copysign

        // Square root (more)
        rw!("sqrt_pow2"; "(pow (sqrt ?x) ?y)" => "(pow ?x (/ ?y 2))"), // sqrt-pow2
        rw!("sqrt_unprod"; "(* (sqrt ?x) (sqrt ?y))" => "(sqrt (* ?x ?y))"), // sqrt-unprod
        rw!("sqrt_undiv"; "(/ (sqrt ?x) (sqrt ?y))" => "(sqrt (/ ?x ?y))"), // sqrt-undiv
        rw!("sqrt_prod"; "(sqrt (* ?x ?y))" => "(* (sqrt (fabs ?x)) (sqrt (fabs ?y)))"), // sqrt-prod
        rw!("sqrt_div"; "(sqrt (/ ?x ?y))" => "(/ (sqrt (fabs ?x)) (sqrt (fabs ?y)))"),  // sqrt-div
        // The next rule uses 'copysign' and is commented out.
        // rw!("add_sqr_sqrt"; "?x" => "(copysign (* (sqrt (fabs ?x)) (sqrt (fabs ?x))) ?x)"), // add-sqr-sqrt

        // Cubing
        rw!("rem_cube_cbrt"; "(pow (cbrt ?x) 3)" => "?x"), // rem-cube-cbrt
        rw!("rem_cbrt_cube"; "(cbrt (pow ?x 3))" => "?x"), // rem-cbrt-cube
        rw!("rem_3cbrt_lft"; "(* (* (cbrt ?x) (cbrt ?x)) (cbrt ?x))" => "?x"), // rem-3cbrt-lft
        rw!("rem_3cbrt_rft"; "(* (cbrt ?x) (* (cbrt ?x) (cbrt ?x)))" => "?x"), // rem-3cbrt-rft
        rw!("cube_neg"; "(pow (neg ?x) 3)" => "(neg (pow ?x 3))"), // cube-neg
        rw!("cube_neg_rev"; "(neg (pow ?x 3))" => "(pow (neg ?x) 3)"), // cube-neg-rev
        rw!("cube_prod"; "(pow (* ?x ?y) 3)" => "(* (pow ?x 3) (pow ?y 3))"), // cube-prod
        rw!("cube_div"; "(pow (/ ?x ?y) 3)" => "(/ (pow ?x 3) (pow ?y 3))"), // cube-div
        rw!("cube_mult"; "(pow ?x 3)" => "(* ?x (* ?x ?x))"), // cube-mult
        rw!("cube_prod_rev"; "(* (pow ?x 3) (pow ?y 3))" => "(pow (* ?x ?y) 3)"), // cube-prod-rev
        rw!("cube_div_rev"; "(/ (pow ?x 3) (pow ?y 3))" => "(pow (/ ?x ?y) 3)"), // cube-div-rev
        // Cube root
        rw!("cbrt_prod"; "(cbrt (* ?x ?y))" => "(* (cbrt ?x) (cbrt ?y))"), // cbrt-prod
        rw!("cbrt_div"; "(cbrt (/ ?x ?y))" => "(/ (cbrt ?x) (cbrt ?y))"),  // cbrt-div
        rw!("cbrt_unprod"; "(* (cbrt ?x) (cbrt ?y))" => "(cbrt (* ?x ?y))"), // cbrt-unprod
        rw!("cbrt_undiv"; "(/ (cbrt ?x) (cbrt ?y))" => "(cbrt (/ ?x ?y))"), // cbrt-undiv
        rw!("pow_cbrt"; "(pow (cbrt ?x) ?y)" => "(pow ?x (/ ?y 3))"),      // pow-cbrt
        rw!("cbrt_pow"; "(cbrt (pow ?x ?y))" => "(pow ?x (/ ?y 3))"),      // cbrt-pow
        rw!("add_cube_cbrt"; "?x" => "(* (* (cbrt ?x) (cbrt ?x)) (cbrt ?x))"), // add-cube-cbrt
        rw!("add_cbrt_cube"; "?x" => "(cbrt (* (* ?x ?x) ?x))"),           // add-cbrt-cube
        rw!("cube_unmult"; "(* ?x (* ?x ?x))" => "(pow ?x 3)"),            // cube-unmult
        rw!("cbrt_neg"; "(cbrt (neg ?x))" => "(neg (cbrt ?x))"),           // cbrt-neg
        rw!("cbrt_neg_rev"; "(neg (cbrt ?x))" => "(cbrt (neg ?x))"),       // cbrt-neg-rev
        rw!("cbrt_fabs"; "(cbrt (fabs ?x))" => "(fabs (cbrt ?x))"),        // cbrt-fabs
        rw!("cbrt_fabs_rev"; "(fabs (cbrt ?x))" => "(cbrt (fabs ?x))"),    // cbrt-fabs-rev
        // The next two use 'copysign' and are commented out.
        // rw!("cbrt_div_cbrt"; "(/ (cbrt ?x) (fabs (cbrt ?x)))" => "(copysign 1 ?x)"), // cbrt-div-cbrt
        // rw!("cbrt_div_cbrt2"; "(/ (fabs (cbrt ?x)) (cbrt ?x))" => "(copysign 1 ?x)"), // cbrt-div-cbrt2

        // Min and max (not in Math enum, commented out)
        // rw!("fmin_swap"; "(fmin ?a ?b)" => "(fmin ?b ?a)"), // fmin-swap
        // rw!("fmax_swap"; "(fmax ?a ?b)" => "(fmax ?b ?a)"), // fmax-swap

        // Exponentials
        rw!("add_log_exp"; "?x" => "(log (exp ?x))"), // add-log-exp
        // rw!("add_exp_log"; "?x" => "(exp (log ?x))"), // add-exp-log #:unsound
        rw!("rem_exp_log"; "(exp (log ?x))" => "?x"), // rem-exp-log
        rw!("rem_log_exp"; "(log (exp ?x))" => "?x"), // rem-log-exp
        // Exponential constants
        rw!("exp_0"; "(exp 0)" => "1"),                   // exp-0
        rw!("exp_1_e"; "(exp 1)" => "(e)"),               // exp-1-e
        rw!("one_exp"; "1" => "(exp 0)"),                 // 1-exp
        rw!("e_exp_1"; "(e)" => "(exp 1)"),               // e-exp-1
        rw!("exp_fabs"; "(exp ?x)" => "(fabs (exp ?x))"), // exp-fabs
        rw!("fabs_exp"; "(fabs (exp ?x))" => "(exp ?x)"), // fabs-exp
        // Exponential identities
        rw!("exp_sum"; "(exp (+ ?a ?b))" => "(* (exp ?a) (exp ?b))"), // exp-sum
        rw!("exp_neg"; "(exp (neg ?a))" => "(/ 1 (exp ?a))"),         // exp-neg
        rw!("exp_diff"; "(exp (- ?a ?b))" => "(/ (exp ?a) (exp ?b))"), // exp-diff
        rw!("prod_exp"; "(* (exp ?a) (exp ?b))" => "(exp (+ ?a ?b))"), // prod-exp
        rw!("rec_exp"; "(/ 1 (exp ?a))" => "(exp (neg ?a))"),         // rec-exp
        rw!("div_exp"; "(/ (exp ?a) (exp ?b))" => "(exp (- ?a ?b))"), // div-exp
        rw!("exp_prod"; "(exp (* ?a ?b))" => "(pow (exp ?a) ?b)"),    // exp-prod
        rw!("exp_sqrt"; "(exp (/ ?a 2))" => "(sqrt (exp ?a))"),       // exp-sqrt
        rw!("exp_cbrt"; "(exp (/ ?a 3))" => "(cbrt (exp ?a))"),       // exp-cbrt
        rw!("exp_lft_sqr"; "(exp (* ?a 2))" => "(* (exp ?a) (exp ?a))"), // exp-lft-sqr
        rw!("exp_lft_cube"; "(exp (* ?a 3))" => "(pow (exp ?a) 3)"),  // exp-lft-cube
        rw!("exp_cbrt_rev"; "(cbrt (exp ?a))" => "(exp (/ ?a 3))"),   // exp-cbrt-rev
        rw!("exp_lft_cube_rev"; "(pow (exp ?a) 3)" => "(exp (* ?a 3))"), // exp-lft-cube-rev
        rw!("exp_sqrt_rev"; "(sqrt (exp ?a))" => "(exp (/ ?a 2))"),   // exp-sqrt-rev
        rw!("exp_lft_sqr_rev"; "(* (exp ?a) (exp ?a))" => "(exp (* ?a 2))"), // exp-lft-sqr-rev
        // Powers
        rw!("unpow_neg1"; "(pow ?a -1)" => "(/ 1 ?a)"), // unpow-1
        rw!("unpow1"; "(pow ?a 1)" => "?a"),            // unpow1
        rw!("unpow0"; "(pow ?a 0)" => "1"),             // unpow0
        rw!("pow_base_1"; "(pow 1 ?a)" => "1"),         // pow-base-1
        rw!("pow1"; "?a" => "(pow ?a 1)"),              // pow1
        rw!("unpow_half"; "(pow ?a 1/2)" => "(sqrt ?a)"), // unpow1/2
        rw!("unpow2"; "(pow ?a 2)" => "(* ?a ?a)"),     // unpow2
        rw!("unpow3"; "(pow ?a 3)" => "(* (* ?a ?a) ?a)"), // unpow3
        rw!("unpow_third"; "(pow ?a 1/3)" => "(cbrt ?a)"), // unpow1/3
        rw!("pow_base_0"; "(pow 0 ?a)" => "0"),         // pow-base-0
        rw!("inv_pow"; "(/ 1 ?a)" => "(pow ?a -1)"),    // inv-pow
        rw!("pow_half"; "(sqrt ?a)" => "(pow ?a 1/2)"), // pow1/2
        rw!("pow2"; "(* ?a ?a)" => "(pow ?a 2)"),       // pow2
        rw!("pow_third"; "(cbrt ?a)" => "(pow ?a 1/3)"), // pow1/3
        rw!("pow3"; "(* (* ?a ?a) ?a)" => "(pow ?a 3)"), // pow3
        rw!("exp_to_pow"; "(exp (* (log ?a) ?b))" => "(pow ?a ?b)"), // exp-to-pow
        rw!("pow_plus"; "(* (pow ?a ?b) ?a)" => "(pow ?a (+ ?b 1))"), // pow-plus
        rw!("pow_exp"; "(pow (exp ?a) ?b)" => "(exp (* ?a ?b))"), // pow-exp
        rw!("pow_prod_down"; "(* (pow ?b ?a) (pow ?c ?a))" => "(pow (* ?b ?c) ?a)"), // pow-prod-down
        rw!("pow_prod_up"; "(* (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (+ ?b ?c))"),   // pow-prod-up
        rw!("pow_flip"; "(/ 1 (pow ?a ?b))" => "(pow ?a (neg ?b))"),                 // pow-flip
        rw!("pow_div"; "(/ (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (- ?b ?c))"),       // pow-div
        // The following use sound-pow and are commented out.
        // rw!("pow_plus_rev"; "(pow ?a (+ ?b 1))" => "(* (sound-pow ?a ?b 1) ?a)"), // pow-plus-rev
        // rw!("pow_neg"; "(pow ?a (neg ?b))" => "(sound-/ 1 (sound-pow ?a ?b 0) 0)"), // pow-neg

        // Unsound power rules (commented out)
        // rw!("pow_to_exp"; "(pow ?a ?b)" => "(exp (* (log ?a) ?b))"), // pow-to-exp #:unsound
        // rw!("pow_add"; "(pow ?a (+ ?b ?c))" => "(* (pow ?a ?b) (pow ?a ?c))"), // pow-add #:unsound
        // rw!("pow_sub"; "(pow ?a (- ?b ?c))" => "(/ (pow ?a ?b) (pow ?a ?c))"), // pow-sub #:unsound
        // rw!("unpow_prod_down"; "(pow (* ?b ?c) ?a)" => "(* (pow ?b ?a) (pow ?c ?a))"), // unpow-prod-down #:unsound

        // Logarithms
        rw!("log_rec"; "(log (/ 1 ?a))" => "(neg (log ?a))"), // log-rec
        rw!("log_e"; "(log (e))" => "1"),                     // log-E
        rw!("log_pow_rev"; "(* ?b (log ?a))" => "(log (pow ?a ?b))"), // log-pow-rev
        rw!("log_prod"; "(log (* ?a ?b))" => "(+ (log (fabs ?a)) (log (fabs ?b)))"), // log-prod
        rw!("log_div"; "(log (/ ?a ?b))" => "(- (log (fabs ?a)) (log (fabs ?b)))"), // log-div
        // rw!("log_pow"; "(log (pow ?a ?b))" => "(* ?b (sound-log (fabs ?a) 0))"), // log-pow (requires sound-log)
        rw!("sum_log"; "(+ (log ?a) (log ?b))" => "(log (* ?a ?b))"), // sum-log
        rw!("diff_log"; "(- (log ?a) (log ?b))" => "(log (/ ?a ?b))"), // diff-log
        rw!("neg_log"; "(neg (log ?a))" => "(log (/ 1 ?a))"),         // neg-log
        // Trigonometry (basic)
        rw!("sin_0"; "(sin 0)" => "0"),                       // sin-0
        rw!("cos_0"; "(cos 0)" => "1"),                       // cos-0
        rw!("tan_0"; "(tan 0)" => "0"),                       // tan-0
        rw!("sin_neg"; "(sin (neg ?x))" => "(neg (sin ?x))"), // sin-neg
        rw!("cos_neg"; "(cos (neg ?x))" => "(cos ?x)"),       // cos-neg
        rw!("cos_fabs"; "(cos (fabs ?x))" => "(cos ?x)"),     // cos-fabs
        rw!("tan_neg"; "(tan (neg ?x))" => "(neg (tan ?x))"), // tan-neg
        rw!("cos_neg_rev"; "(cos ?x)" => "(cos (neg ?x))"),   // cos-neg-rev
        rw!("cos_fabs_rev"; "(cos ?x)" => "(cos (fabs ?x))"), // cos-fabs-rev
        rw!("sin_neg_rev"; "(neg (sin ?x))" => "(sin (neg ?x))"), // sin-neg-rev
        rw!("tan_neg_rev"; "(neg (tan ?x))" => "(tan (neg ?x))"), // tan-neg-rev
        // Trig identities (Pythagorean)
        rw!("sqr_sin_b"; "(* (sin ?x) (sin ?x))" => "(- 1 (* (cos ?x) (cos ?x)))"), // sqr-sin-b
        rw!("sqr_cos_b"; "(* (cos ?x) (cos ?x))" => "(- 1 (* (sin ?x) (sin ?x)))"), // sqr-cos-b
        rw!("sqr_cos_b_rev"; "(- 1 (* (sin ?x) (sin ?x)))" => "(* (cos ?x) (cos ?x))"), // sqr-cos-b-rev
        rw!("sqr_sin_b_rev"; "(- 1 (* (cos ?x) (cos ?x)))" => "(* (sin ?x) (sin ?x))"), // sqr-sin-b-rev
        // Inverse trig
        rw!("sin_asin"; "(sin (asin ?x))" => "?x"), // sin-asin
        rw!("cos_acos"; "(cos (acos ?x))" => "?x"), // cos-acos
        rw!("tan_atan"; "(tan (atan ?x))" => "?x"), // tan-atan
        // The following use 'PI' and 'remainder', not in language; commented out.
        rw!("atan_tan"; "(atan (tan ?x))" => "(remainder ?x (pi))"), // atan-tan
        rw!("asin_sin"; "(asin (sin ?x))" => "(- (fabs (remainder (+ ?x (/ (pi) 2)) (* 2 (pi)))) (/ (pi) 2))"), // asin-sin
        rw!("acos_cos"; "(acos (cos ?x))" => "(fabs (remainder ?x (* 2 (pi))))"), // acos-cos
        rw!("acos_cos_rev"; "(fabs (remainder ?x (* 2 (pi))))" => "(acos (cos ?x))"), // acos-cos-rev
        rw!("asin_sin_rev"; "(- (fabs (remainder (+ ?x (/ (pi) 2)) (* 2 (pi)))) (/ (pi) 2))" => "(asin (sin ?x))"), // asin-sin-rev
        // More trig identities
        rw!("cos_sin_sum"; "(+ (* (cos ?a) (cos ?a)) (* (sin ?a) (sin ?a)))" => "1"), // cos-sin-sum
        rw!("one_sub_cos"; "(- 1 (* (cos ?a) (cos ?a)))" => "(* (sin ?a) (sin ?a))"), // 1-sub-cos
        rw!("one_sub_sin"; "(- 1 (* (sin ?a) (sin ?a)))" => "(* (cos ?a) (cos ?a))"), // 1-sub-sin
        rw!("neg1_add_cos"; "(+ (* (cos ?a) (cos ?a)) -1)" => "(neg (* (sin ?a) (sin ?a)))"), // -1-add-cos
        rw!("neg1_add_sin"; "(+ (* (sin ?a) (sin ?a)) -1)" => "(neg (* (cos ?a) (cos ?a)))"), // -1-add-sin
        rw!("sub1_cos"; "(- (* (cos ?a) (cos ?a)) 1)" => "(neg (* (sin ?a) (sin ?a)))"), // sub-1-cos
        rw!("sub1_sin"; "(- (* (sin ?a) (sin ?a)) 1)" => "(neg (* (cos ?a) (cos ?a)))"), // sub-1-sin
        rw!("sin_pi_6"; "(sin (/ (pi) 6))" => "1/2"),                                    // sin-PI/6
        rw!("sin_pi_4"; "(sin (/ (pi) 4))" => "(/ (sqrt 2) 2)"),                         // sin-PI/4
        rw!("sin_pi_3"; "(sin (/ (pi) 3))" => "(/ (sqrt 3) 2)"),                         // sin-PI/3
        rw!("sin_pi_2"; "(sin (/ (pi) 2))" => "1"),                                      // sin-PI/2
        rw!("sin_pi"; "(sin (pi))" => "0"),                                              // sin-PI
        rw!("sin_add_pi"; "(sin (+ ?x (pi)))" => "(neg (sin ?x))"),                      // sin-+PI
        rw!("sin_add_pi_2"; "(sin (+ ?x (/ (pi) 2)))" => "(cos ?x)"), // sin-+PI/2
        rw!("cos_pi_6"; "(cos (/ (pi) 6))" => "(/ (sqrt 3) 2)"),      // cos-PI/6
        rw!("cos_pi_4"; "(cos (/ (pi) 4))" => "(/ (sqrt 2) 2)"),      // cos-PI/4
        rw!("cos_pi_3"; "(cos (/ (pi) 3))" => "1/2"),                 // cos-PI/3
        rw!("cos_pi_2"; "(cos (/ (pi) 2))" => "0"),                   // cos-PI/2
        rw!("cos_pi"; "(cos (pi))" => "-1"),                          // cos-PI
        rw!("cos_add_pi"; "(cos (+ ?x (pi)))" => "(neg (cos ?x))"),   // cos-+PI
        rw!("cos_add_pi_2"; "(cos (+ ?x (/ (pi) 2)))" => "(neg (sin ?x))"), // cos-+PI/2
        rw!("tan_pi_6"; "(tan (/ (pi) 6))" => "(/ 1 (sqrt 3))"),      // tan-PI/6
        rw!("tan_pi_4"; "(tan (/ (pi) 4))" => "1"),                   // tan-PI/4
        rw!("tan_pi_3"; "(tan (/ (pi) 3))" => "(sqrt 3)"),            // tan-PI/3
        rw!("tan_pi"; "(tan (pi))" => "0"),                           // tan-PI
        rw!("tan_add_pi"; "(tan (+ ?x (pi)))" => "(tan ?x)"),         // tan-+PI
        rw!("hang_0p_tan"; "(/ (sin ?a) (+ 1 (cos ?a)))" => "(tan (/ ?a 2))"), // hang-0p-tan
        rw!("hang_0m_tan"; "(/ (neg (sin ?a)) (+ 1 (cos ?a)))" => "(tan (/ (neg ?a) 2))"), // hang-0m-tan
        rw!("hang_p0_tan"; "(/ (- 1 (cos ?a)) (sin ?a))" => "(tan (/ ?a 2))"), // hang-p0-tan
        rw!("hang_m0_tan"; "(/ (- 1 (cos ?a)) (neg (sin ?a)))" => "(tan (/ (neg ?a) 2))"), // hang-m0-tan
        rw!("hang_p_tan"; "(/ (+ (sin ?a) (sin ?b)) (+ (cos ?a) (cos ?b)))" => "(tan (/ (+ ?a ?b) 2))"), // hang-p-tan
        rw!("hang_m_tan"; "(/ (- (sin ?a) (sin ?b)) (+ (cos ?a) (cos ?b)))" => "(tan (/ (- ?a ?b) 2))"), // hang-m-tan
        // Reverse trig identities
        rw!("one_sub_sin_rev"; "(* (cos ?a) (cos ?a))" => "(- 1 (* (sin ?a) (sin ?a)))"), // 1-sub-sin-rev
        rw!("hang_0m_tan_rev"; "(tan (/ (neg ?a) 2))" => "(/ (neg (sin ?a)) (+ 1 (cos ?a)))"), // hang-0m-tan-rev
        rw!("hang_0p_tan_rev"; "(tan (/ ?a 2))" => "(/ (sin ?a) (+ 1 (cos ?a)))"), // hang-0p-tan-rev
        rw!("tan_add_pi_rev"; "(tan ?x)" => "(tan (+ ?x (pi)))"),                  // tan-+PI-rev
        rw!("cos_add_pi_2_rev"; "(neg (sin ?x))" => "(cos (+ ?x (/ (pi) 2)))"),    // cos-+PI/2-rev
        rw!("sin_add_pi_2_rev"; "(cos ?x)" => "(sin (+ ?x (/ (pi) 2)))"),          // sin-+PI/2-rev
        rw!("sin_add_pi_rev"; "(neg (sin ?x))" => "(sin (+ ?x (pi)))"),            // sin-+PI-rev
        rw!("cos_add_pi_rev"; "(neg (cos ?x))" => "(cos (+ ?x (pi)))"),            // cos-+PI-rev
        rw!("neg_tan_add_pi_2_rev"; "(/ -1 (tan ?x))" => "(tan (+ ?x (/ (pi) 2)))"), // neg-tan-+PI/2-rev
        rw!("tan_add_pi_2_rev"; "(/ 1 (tan ?x))" => "(tan (+ (neg ?x) (/ (pi) 2)))"), // tan-+PI/2-rev
        // Angle sum/difference formulas
        rw!("sin_sum"; "(sin (+ ?x ?y))" => "(+ (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))"), // sin-sum
        rw!("cos_sum"; "(cos (+ ?x ?y))" => "(- (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))"), // cos-sum
        rw!("sin_diff"; "(sin (- ?x ?y))" => "(- (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))"), // sin-diff
        rw!("cos_diff"; "(cos (- ?x ?y))" => "(+ (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))"), // cos-diff
        rw!("sin_2"; "(sin (* 2 ?x))" => "(* 2 (* (sin ?x) (cos ?x)))"), // sin-2
        rw!("sin_3"; "(sin (* 3 ?x))" => "(- (* 3 (sin ?x)) (* 4 (pow (sin ?x) 3)))"), // sin-3
        rw!("two_sin"; "(* 2 (* (sin ?x) (cos ?x)))" => "(sin (* 2 ?x))"), // 2-sin
        rw!("three_sin"; "(- (* 3 (sin ?x)) (* 4 (pow (sin ?x) 3)))" => "(sin (* 3 ?x))"), // 3-sin
        rw!("cos_2"; "(cos (* 2 ?x))" => "(- (* (cos ?x) (cos ?x)) (* (sin ?x) (sin ?x)))"), // cos-2
        rw!("cos_3"; "(cos (* 3 ?x))" => "(- (* 4 (pow (cos ?x) 3)) (* 3 (cos ?x)))"), // cos-3
        rw!("two_cos"; "(- (* (cos ?x) (cos ?x)) (* (sin ?x) (sin ?x)))" => "(cos (* 2 ?x))"), // 2-cos
        rw!("three_cos"; "(- (* 4 (pow (cos ?x) 3)) (* 3 (cos ?x)))" => "(cos (* 3 ?x))"), // 3-cos
        // Reverse angle formulas
        rw!("cos_diff_rev"; "(+ (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))" => "(cos (- ?x ?y))"), // cos-diff-rev
        rw!("sin_diff_rev"; "(- (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))" => "(sin (- ?x ?y))"), // sin-diff-rev
        rw!("sin_sum_rev"; "(+ (* (sin ?x) (cos ?y)) (* (cos ?x) (sin ?y)))" => "(sin (+ ?x ?y))"), // sin-sum-rev
        rw!("tan_sum_rev"; "(/ (+ (tan ?x) (tan ?y)) (- 1 (* (tan ?x) (tan ?y))))" => "(tan (+ ?x ?y))"), // tan-sum-rev
        rw!("cos_sum_rev"; "(- (* (cos ?x) (cos ?y)) (* (sin ?x) (sin ?y)))" => "(cos (+ ?x ?y))"), // cos-sum-rev
        // More trig identities
        rw!("sqr_sin_a"; "(* (sin ?x) (sin ?x))" => "(- 1/2 (* 1/2 (cos (* 2 ?x))))"), // sqr-sin-a
        rw!("sqr_cos_a"; "(* (cos ?x) (cos ?x))" => "(+ 1/2 (* 1/2 (cos (* 2 ?x))))"), // sqr-cos-a
        rw!("diff_sin"; "(- (sin ?x) (sin ?y))" => "(* 2 (* (sin (/ (- ?x ?y) 2)) (cos (/ (+ ?x ?y) 2))))"), // diff-sin
        rw!("diff_cos"; "(- (cos ?x) (cos ?y))" => "(* -2 (* (sin (/ (- ?x ?y) 2)) (sin (/ (+ ?x ?y) 2))))"), // diff-cos
        rw!("sum_sin"; "(+ (sin ?x) (sin ?y))" => "(* 2 (* (sin (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))"), // sum-sin
        rw!("sum_cos"; "(+ (cos ?x) (cos ?y))" => "(* 2 (* (cos (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))"), // sum-cos
        rw!("cos_mult"; "(* (cos ?x) (cos ?y))" => "(/ (+ (cos (+ ?x ?y)) (cos (- ?x ?y))) 2)"), // cos-mult
        rw!("sin_mult"; "(* (sin ?x) (sin ?y))" => "(/ (- (cos (- ?x ?y)) (cos (+ ?x ?y))) 2)"), // sin-mult
        rw!("sin_cos_mult"; "(* (sin ?x) (cos ?y))" => "(/ (+ (sin (- ?x ?y)) (sin (+ ?x ?y))) 2)"), // sin-cos-mult
        rw!("diff_atan"; "(- (atan ?x) (atan ?y))" => "(atan2 (- ?x ?y) (+ 1 (* ?x ?y)))"), // diff-atan
        rw!("sum_atan"; "(+ (atan ?x) (atan ?y))" => "(atan2 (+ ?x ?y) (- 1 (* ?x ?y)))"), // sum-atan
        rw!("tan_quot"; "(tan ?x)" => "(/ (sin ?x) (cos ?x))"), // tan-quot
        rw!("quot_tan"; "(/ (sin ?x) (cos ?x))" => "(tan ?x)"), // quot-tan
        rw!("two_tan"; "(/ (* 2 (tan ?x)) (- 1 (* (tan ?x) (tan ?x))))" => "(tan (* 2 ?x))"), // 2-tan
        // Reverse trig more
        rw!("diff_cos_rev"; "(* -2 (* (sin (/ (- ?x ?y) 2)) (sin (/ (+ ?x ?y) 2))))" => "(- (cos ?x) (cos ?y))"), // diff-cos-rev
        rw!("diff_sin_rev"; "(* 2 (* (sin (/ (- ?x ?y) 2)) (cos (/ (+ ?x ?y) 2))))" => "(- (sin ?x) (sin ?y))"), // diff-sin-rev
        rw!("diff_atan_rev"; "(atan2 (- ?x ?y) (+ 1 (* ?x ?y)))" => "(- (atan ?x) (atan ?y))"), // diff-atan-rev
        rw!("sum_sin_rev"; "(* 2 (* (sin (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))" => "(+ (sin ?x) (sin ?y))"), // sum-sin-rev
        rw!("sum_cos_rev"; "(* 2 (* (cos (/ (+ ?x ?y) 2)) (cos (/ (- ?x ?y) 2))))" => "(+ (cos ?x) (cos ?y))"), // sum-cos-rev
        rw!("sum_atan_rev"; "(atan2 (+ ?x ?y) (- 1 (* ?x ?y)))" => "(+ (atan ?x) (atan ?y))"), // sum-atan-rev
        rw!("sqr_cos_a_rev"; "(+ 1/2 (* 1/2 (cos (* 2 ?x))))" => "(* (cos ?x) (cos ?x))"), // sqr-cos-a-rev
        rw!("sqr_sin_a_rev"; "(- 1/2 (* 1/2 (cos (* 2 ?x))))" => "(* (sin ?x) (sin ?x))"), // sqr-sin-a-rev
        rw!("cos_mult_rev"; "(/ (+ (cos (+ ?x ?y)) (cos (- ?x ?y))) 2)" => "(* (cos ?x) (cos ?y))"), // cos-mult-rev
        rw!("sin_mult_rev"; "(/ (- (cos (- ?x ?y)) (cos (+ ?x ?y))) 2)" => "(* (sin ?x) (sin ?y))"), // sin-mult-rev
        rw!("sin_cos_mult_rev"; "(/ (+ (sin (- ?x ?y)) (sin (+ ?x ?y))) 2)" => "(* (sin ?x) (cos ?y))"), // sin-cos-mult-rev
        // Inverse trig compositions
        rw!("cos_asin"; "(cos (asin ?x))" => "(sqrt (- 1 (* ?x ?x)))"), // cos-asin
        rw!("tan_asin"; "(tan (asin ?x))" => "(/ ?x (sqrt (- 1 (* ?x ?x))))"), // tan-asin
        rw!("sin_acos"; "(sin (acos ?x))" => "(sqrt (- 1 (* ?x ?x)))"), // sin-acos
        rw!("tan_acos"; "(tan (acos ?x))" => "(/ (sqrt (- 1 (* ?x ?x))) ?x)"), // tan-acos
        rw!("sin_atan"; "(sin (atan ?x))" => "(/ ?x (sqrt (+ 1 (* ?x ?x))))"), // sin-atan
        rw!("cos_atan"; "(cos (atan ?x))" => "(/ 1 (sqrt (+ 1 (* ?x ?x))))"), // cos-atan
        rw!("asin_acos"; "(asin ?x)" => "(- (/ (pi) 2) (acos ?x))"),    // asin-acos
        rw!("acos_asin"; "(acos ?x)" => "(- (/ (pi) 2) (asin ?x))"),    // acos-asin
        rw!("asin_neg"; "(asin (neg ?x))" => "(neg (asin ?x))"),        // asin-neg
        rw!("acos_neg"; "(acos (neg ?x))" => "(- (pi) (acos ?x))"),     // acos-neg
        rw!("atan_neg"; "(atan (neg ?x))" => "(neg (atan ?x))"),        // atan-neg
        // Reverse inverse trig
        rw!("acos_asin_rev"; "(- (/ (pi) 2) (asin ?x))" => "(acos ?x)"), // acos-asin-rev
        rw!("asin_acos_rev"; "(- (/ (pi) 2) (acos ?x))" => "(asin ?x)"), // asin-acos-rev
        rw!("asin_neg_rev"; "(neg (asin ?x))" => "(asin (neg ?x))"),     // asin-neg-rev
        rw!("atan_neg_rev"; "(neg (atan ?x))" => "(atan (neg ?x))"),     // atan-neg-rev
        rw!("acos_neg_rev"; "(- (pi) (acos ?x))" => "(acos (neg ?x))"),  // acos-neg-rev
        rw!("cos_atan_rev"; "(/ 1 (sqrt (+ 1 (* ?x ?x))))" => "(cos (atan ?x))"), // cos-atan-rev
        rw!("tan_acos_rev"; "(/ (sqrt (- 1 (* ?x ?x))) ?x)" => "(tan (acos ?x))"), // tan-acos-rev
        rw!("tan_asin_rev"; "(/ ?x (sqrt (- 1 (* ?x ?x))))" => "(tan (asin ?x))"), // tan-asin-rev
        rw!("cos_asin_rev"; "(sqrt (- 1 (* ?x ?x)))" => "(cos (asin ?x))"), // cos-asin-rev
        rw!("sin_atan_rev"; "(/ ?x (sqrt (+ 1 (* ?x ?x))))" => "(sin (atan ?x))"), // sin-atan-rev
        rw!("sin_acos_rev"; "(sqrt (- 1 (* ?x ?x)))" => "(sin (acos ?x))"), // sin-acos-rev
        // Hyperbolic definitions
        rw!("sinh_def"; "(sinh ?x)" => "(/ (- (exp ?x) (exp (neg ?x))) 2)"), // sinh-def
        rw!("cosh_def"; "(cosh ?x)" => "(/ (+ (exp ?x) (exp (neg ?x))) 2)"), // cosh-def
        rw!("tanh_def_a"; "(tanh ?x)" => "(/ (- (exp ?x) (exp (neg ?x))) (+ (exp ?x) (exp (neg ?x))))"), // tanh-def-a
        rw!("tanh_def_b"; "(tanh ?x)" => "(/ (- (exp (* 2 ?x)) 1) (+ (exp (* 2 ?x)) 1))"), // tanh-def-b
        rw!("tanh_def_c"; "(tanh ?x)" => "(/ (- 1 (exp (* -2 ?x))) (+ 1 (exp (* -2 ?x))))"), // tanh-def-c
        rw!("sinh_cosh"; "(- (* (cosh ?x) (cosh ?x)) (* (sinh ?x) (sinh ?x)))" => "1"), // sinh-cosh
        rw!("sinh_plus_cosh"; "(+ (cosh ?x) (sinh ?x))" => "(exp ?x)"), // sinh-+-cosh
        rw!("sinh_minus_cosh"; "(- (cosh ?x) (sinh ?x))" => "(exp (neg ?x))"), // sinh---cosh
        // Hyperbolic reverses
        rw!("tanh_def_b_rev"; "(/ (- (exp (* 2 ?x)) 1) (+ (exp (* 2 ?x)) 1))" => "(tanh ?x)"), // tanh-def-b-rev
        rw!("tanh_def_c_rev"; "(/ (- 1 (exp (* -2 ?x))) (+ 1 (exp (* -2 ?x))))" => "(tanh ?x)"), // tanh-def-c-rev
        rw!("sinh_def_rev"; "(/ (- (exp ?x) (exp (neg ?x))) 2)" => "(sinh ?x)"), // sinh-def-rev
        rw!("cosh_def_rev"; "(/ (+ (exp ?x) (exp (neg ?x))) 2)" => "(cosh ?x)"), // cosh-def-rev
        rw!("sinh_plus_cosh_rev"; "(exp ?x)" => "(+ (cosh ?x) (sinh ?x))"),      // sinh-+-cosh-rev
        rw!("sinh_minus_cosh_rev"; "(exp (neg ?x))" => "(- (cosh ?x) (sinh ?x))"), // sinh---cosh-rev
        // Hyperbolic identities
        rw!("sinh_undef"; "(- (exp ?x) (exp (neg ?x)))" => "(* 2 (sinh ?x))"), // sinh-undef
        rw!("cosh_undef"; "(+ (exp ?x) (exp (neg ?x)))" => "(* 2 (cosh ?x))"), // cosh-undef
        rw!("tanh_undef"; "(/ (- (exp ?x) (exp (neg ?x))) (+ (exp ?x) (exp (neg ?x))))" => "(tanh ?x)"), // tanh-undef
        rw!("cosh_sum"; "(cosh (+ ?x ?y))" => "(+ (* (cosh ?x) (cosh ?y)) (* (sinh ?x) (sinh ?y)))"), // cosh-sum
        rw!("cosh_diff"; "(cosh (- ?x ?y))" => "(- (* (cosh ?x) (cosh ?y)) (* (sinh ?x) (sinh ?y)))"), // cosh-diff
        rw!("cosh_2"; "(cosh (* 2 ?x))" => "(+ (* (sinh ?x) (sinh ?x)) (* (cosh ?x) (cosh ?x)))"), // cosh-2
        rw!("cosh_half"; "(cosh (/ ?x 2))" => "(sqrt (/ (+ (cosh ?x) 1) 2))"), // cosh-1/2
        rw!("sinh_sum"; "(sinh (+ ?x ?y))" => "(+ (* (sinh ?x) (cosh ?y)) (* (cosh ?x) (sinh ?y)))"), // sinh-sum
        rw!("sinh_diff"; "(sinh (- ?x ?y))" => "(- (* (sinh ?x) (cosh ?y)) (* (cosh ?x) (sinh ?y)))"), // sinh-diff
        rw!("sinh_2"; "(sinh (* 2 ?x))" => "(* 2 (* (sinh ?x) (cosh ?x)))"), // sinh-2
        rw!("sinh_half"; "(sinh (/ ?x 2))" => "(/ (sinh ?x) (sqrt (* 2 (+ (cosh ?x) 1))))"), // sinh-1/2
        rw!("tanh_2"; "(tanh (* 2 ?x))" => "(/ (* 2 (tanh ?x)) (+ 1 (* (tanh ?x) (tanh ?x))))"), // tanh-2
        rw!("tanh_half"; "(tanh (/ ?x 2))" => "(/ (sinh ?x) (+ (cosh ?x) 1))"), // tanh-1/2
        rw!("sum_sinh"; "(+ (sinh ?x) (sinh ?y))" => "(* 2 (* (sinh (/ (+ ?x ?y) 2)) (cosh (/ (- ?x ?y) 2))))"), // sum-sinh
        rw!("sum_cosh"; "(+ (cosh ?x) (cosh ?y))" => "(* 2 (* (cosh (/ (+ ?x ?y) 2)) (cosh (/ (- ?x ?y) 2))))"), // sum-cosh
        rw!("diff_sinh"; "(- (sinh ?x) (sinh ?y))" => "(* 2 (* (cosh (/ (+ ?x ?y) 2)) (sinh (/ (- ?x ?y) 2))))"), // diff-sinh
        rw!("diff_cosh"; "(- (cosh ?x) (cosh ?y))" => "(* 2 (* (sinh (/ (+ ?x ?y) 2)) (sinh (/ (- ?x ?y) 2))))"), // diff-cosh
        rw!("tanh_sum"; "(tanh (+ ?x ?y))" => "(/ (+ (tanh ?x) (tanh ?y)) (+ 1 (* (tanh ?x) (tanh ?y))))"), // tanh-sum
        // Hyperbolic reverses (continued)
        rw!("sinh_undef_rev"; "(* 2 (sinh ?x))" => "(- (exp ?x) (exp (neg ?x)))"), // sinh-undef-rev
        rw!("cosh_undef_rev"; "(* 2 (cosh ?x))" => "(+ (exp ?x) (exp (neg ?x)))"), // cosh-undef-rev
        rw!("diff_cosh_rev"; "(* 2 (* (sinh (/ (+ ?x ?y) 2)) (sinh (/ (- ?x ?y) 2))))" => "(- (cosh ?x) (cosh ?y))"), // diff-cosh-rev
        rw!("diff_sinh_rev"; "(* 2 (* (cosh (/ (+ ?x ?y) 2)) (sinh (/ (- ?x ?y) 2))))" => "(- (sinh ?x) (sinh ?y))"), // diff-sinh-rev
        rw!("cosh_diff_rev"; "(- (* (cosh ?x) (cosh ?y)) (* (sinh ?x) (sinh ?y)))" => "(cosh (- ?x ?y))"), // cosh-diff-rev
        rw!("sinh_diff_rev"; "(- (* (sinh ?x) (cosh ?y)) (* (cosh ?x) (sinh ?y)))" => "(sinh (- ?x ?y))"), // sinh-diff-rev
        rw!("tanh_half_rev"; "(/ (sinh ?x) (+ (cosh ?x) 1))" => "(tanh (/ ?x 2))"), // tanh-1/2-rev
        // The next rule appears to be missing from original but was in rev list; included as written.
        // rw!("tanh_half_star_rev"; "(/ (- (cosh ?x) 1) (sinh ?x))" => "(tanh (/ ?x 2))"), // tanh-1/2*-rev
        rw!("tanh_2_rev"; "(/ (* 2 (tanh ?x)) (+ 1 (* (tanh ?x) (tanh ?x))))" => "(tanh (* 2 ?x))"), // tanh-2-rev
        rw!("sinh_half_rev"; "(/ (sinh ?x) (sqrt (* 2 (+ (cosh ?x) 1))))" => "(sinh (/ ?x 2))"), // sinh-1/2-rev
        rw!("cosh_half_rev"; "(sqrt (/ (+ (cosh ?x) 1) 2))" => "(cosh (/ ?x 2))"), // cosh-1/2-rev
        rw!("sinh_2_rev"; "(* 2 (* (sinh ?x) (cosh ?x)))" => "(sinh (* 2 ?x))"),   // sinh-2-rev
        rw!("cosh_2_rev"; "(+ (* (sinh ?x) (sinh ?x)) (* (cosh ?x) (cosh ?x)))" => "(cosh (* 2 ?x))"), // cosh-2-rev
        rw!("sinh_sum_rev"; "(+ (* (sinh ?x) (cosh ?y)) (* (cosh ?x) (sinh ?y)))" => "(sinh (+ ?x ?y))"), // sinh-sum-rev
        rw!("tanh_sum_rev"; "(/ (+ (tanh ?x) (tanh ?y)) (+ 1 (* (tanh ?x) (tanh ?y))))" => "(tanh (+ ?x ?y))"), // tanh-sum-rev
        rw!("cosh_sum_rev"; "(+ (* (cosh ?x) (cosh ?y)) (* (sinh ?x) (sinh ?y)))" => "(cosh (+ ?x ?y))"), // cosh-sum-rev
        rw!("sum_cosh_rev"; "(* 2 (* (cosh (/ (+ ?x ?y) 2)) (cosh (/ (- ?x ?y) 2))))" => "(+ (cosh ?x) (cosh ?y))"), // sum-cosh-rev
        rw!("sum_sinh_rev"; "(* 2 (* (sinh (/ (+ ?x ?y) 2)) (cosh (/ (- ?x ?y) 2))))" => "(+ (sinh ?x) (sinh ?y))"), // sum-sinh-rev
        // Hyperbolic basics
        rw!("sinh_neg"; "(sinh (neg ?x))" => "(neg (sinh ?x))"), // sinh-neg
        rw!("sinh_0"; "(sinh 0)" => "0"),                        // sinh-0
        rw!("sinh_0_rev"; "0" => "(sinh 0)"),                    // sinh-0-rev
        rw!("cosh_neg"; "(cosh (neg ?x))" => "(cosh ?x)"),       // cosh-neg
        rw!("cosh_0"; "(cosh 0)" => "1"),                        // cosh-0
        rw!("cosh_0_rev"; "1" => "(cosh 0)"),                    // cosh-0-rev
        rw!("cosh_neg_rev"; "(cosh ?x)" => "(cosh (neg ?x))"),   // cosh-neg-rev
        rw!("sinh_neg_rev"; "(neg (sinh ?x))" => "(sinh (neg ?x))"), // sinh-neg-rev
        // Inverse hyperbolic definitions
        rw!("asinh_def"; "(asinh ?x)" => "(log (+ ?x (sqrt (+ (* ?x ?x) 1))))"), // asinh-def
        rw!("acosh_def"; "(acosh ?x)" => "(log (+ ?x (sqrt (- (* ?x ?x) 1))))"), // acosh-def
        rw!("atanh_def"; "(atanh ?x)" => "(/ (log (/ (+ 1 ?x) (- 1 ?x))) 2)"),   // atanh-def
        rw!("sinh_asinh"; "(sinh (asinh ?x))" => "?x"),                          // sinh-asinh
        rw!("sinh_acosh"; "(sinh (acosh ?x))" => "(sqrt (- (* ?x ?x) 1))"),      // sinh-acosh
        rw!("sinh_atanh"; "(sinh (atanh ?x))" => "(/ ?x (sqrt (- 1 (* ?x ?x))))"), // sinh-atanh
        rw!("cosh_asinh"; "(cosh (asinh ?x))" => "(sqrt (+ (* ?x ?x) 1))"),      // cosh-asinh
        rw!("cosh_acosh"; "(cosh (acosh ?x))" => "?x"),                          // cosh-acosh
        rw!("cosh_atanh"; "(cosh (atanh ?x))" => "(/ 1 (sqrt (- 1 (* ?x ?x))))"), // cosh-atanh
        rw!("tanh_asinh"; "(tanh (asinh ?x))" => "(/ ?x (sqrt (+ 1 (* ?x ?x))))"), // tanh-asinh
        rw!("tanh_acosh"; "(tanh (acosh ?x))" => "(/ (sqrt (- (* ?x ?x) 1)) ?x)"), // tanh-acosh
        rw!("tanh_atanh"; "(tanh (atanh ?x))" => "?x"),                          // tanh-atanh
        // Inverse hyperbolic reverses
        rw!("asinh_def_rev"; "(log (+ ?x (sqrt (+ (* ?x ?x) 1))))" => "(asinh ?x)"), // asinh-def-rev
        rw!("atanh_def_rev"; "(/ (log (/ (+ 1 ?x) (- 1 ?x))) 2)" => "(atanh ?x)"), // atanh-def-rev
        rw!("acosh_def_rev"; "(log (+ ?x (sqrt (- (* ?x ?x) 1))))" => "(acosh ?x)"), // acosh-def-rev
        rw!("tanh_asinh_rev"; "(/ ?x (sqrt (+ 1 (* ?x ?x))))" => "(tanh (asinh ?x))"), // tanh-asinh-rev
        rw!("cosh_asinh_rev"; "(sqrt (+ (* ?x ?x) 1))" => "(cosh (asinh ?x))"), // cosh-asinh-rev
        rw!("sinh_atanh_rev"; "(/ ?x (sqrt (- 1 (* ?x ?x))))" => "(sinh (atanh ?x))"), // sinh-atanh-rev
        rw!("cosh_atanh_rev"; "(/ 1 (sqrt (- 1 (* ?x ?x))))" => "(cosh (atanh ?x))"), // cosh-atanh-rev
                                                                                      // The next two seem to have mismatched patterns; included as written in source
                                                                                      // rw!("asinh_2"; "(acosh (+ (* 2 (* ?x ?x)) 1))" => "(* 2 (asinh (fabs ?x)))"), // asinh-2
                                                                                      // rw!("acosh_2_rev"; "(* 2 (acosh ?x))" => "(acosh (- (* 2 (* ?x ?x)) 1))"), // acosh-2-rev>
    ]
}
