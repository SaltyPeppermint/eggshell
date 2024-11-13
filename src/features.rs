use crate::trs::TrsLang;

#[must_use]
pub(crate) fn symbol_features<L: TrsLang>(node: &L, variables: usize) -> Vec<f64> {
    let symbols = L::symbols(variables);
    if node.is_const() {
        let mut f = vec![0.0; symbols.len()];
        f.push(1.0);
        f
    } else {
        // account for the const entry
        let mut f = vec![0.0; symbols.len() + 1];
        let p = symbols
            .into_iter()
            .position(|(s, _)| node.to_string() == s)
            .expect("Must be in symbols");
        f[p] = 1.0;
        f
    }

    // Self::symbols(variables)
    //     .into_iter()
    //     .map(|(s, _)| {
    //         if self.to_string() == s && !self.is_const() {
    //             1.0
    //         } else {
    //             0.0
    //         }
    //     })
    //     .chain(if self.is_const() { [1.0] } else { [0.0] })
    //     .chain(additional_features)
    //     .collect()
}

#[must_use]
#[expect(clippy::cast_precision_loss)]
pub(crate) fn additional<L: TrsLang>(node: &L, pos: usize, root_distance: usize) -> Vec<f64> {
    vec![
        pos as f64,
        node.children().len() as f64,
        root_distance as f64,
    ]
}
