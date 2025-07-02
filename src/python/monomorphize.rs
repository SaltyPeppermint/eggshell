/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::Language;
        use egg::RecExpr as EggRecExpr;
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::*;

        use $crate::eqsat::conf::EqsatConf;
        use $crate::eqsat::{Eqsat, StartMaterial};
        use $crate::meta_lang::partial;
        use $crate::meta_lang::{PartialLang, ProbabilisticLang};
        use $crate::python::data::TreeData;
        use $crate::python::err::EggshellError;
        use $crate::trs::{LangExtras, TermRewriteSystem};

        type L = <$type as TermRewriteSystem>::Language;
        // type N = <$type as TermRewriteSystem>::Analysis;

        #[gen_stub_pyclass]
        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct RecExpr(EggRecExpr<L>);

        #[gen_stub_pymethods]
        #[pymethods]
        impl RecExpr {
            /// Parse from string
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(s_expr_str: &str) -> PyResult<RecExpr> {
                let rec_expr = s_expr_str
                    .parse::<egg::RecExpr<L>>()
                    .map_err(|e| EggshellError::from(e))?;
                Ok(RecExpr(rec_expr))
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.0.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[pyo3(signature = (name, path, transparent=false))]
            pub fn to_dot(&self, name: String, path: String, transparent: bool) {
                let dot = $crate::viz::to_dot(&self.0, &name, transparent);
                let svg = $crate::viz::dot_to_svg(&dot);
                let path = std::env::current_dir().unwrap().join(path);
                std::fs::write(path, svg).unwrap();
            }

            #[must_use]
            pub fn distance(&self, other: &RecExpr) -> usize {
                $crate::tree_distance::distance(&self.0, &other.0)
            }

            #[must_use]
            pub fn arity(&self, position: usize) -> usize {
                self.0[egg::Id::from(position)].children().len()
            }

            #[must_use]
            pub fn to_data(&self) -> TreeData {
                (&self.0).into()
            }

            #[staticmethod]
            #[expect(clippy::missing_errors_doc)]
            pub fn from_data(tree_data: TreeData) -> PyResult<RecExpr> {
                let expr = (&tree_data)
                    .try_into()
                    .map_err(|e| EggshellError::<L>::from(e))?;
                Ok(RecExpr(expr))
            }
        }

        #[gen_stub_pyclass]
        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PartialRecExpr {
            expr: EggRecExpr<PartialLang<ProbabilisticLang<L>>>,
            #[pyo3(get)]
            used_tokens: usize,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl PartialRecExpr {
            #[expect(clippy::missing_errors_doc)]
            #[new]
            #[pyo3(signature = (token_list, token_probs=None))]
            pub fn new(
                token_list: Vec<String>,
                token_probs: Option<Vec<f64>>,
            ) -> PyResult<PartialRecExpr> {
                let (partial_node, used_tokens) = partial::partial_parse::<L, _>(
                    token_list.as_slice(),
                    token_probs.as_ref().map(|v| &**v),
                )?;

                Ok(PartialRecExpr {
                    expr: partial_node.into(),
                    used_tokens,
                })
            }

            #[must_use]
            pub fn to_data(&self) -> TreeData {
                (&self.expr).into()
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.expr.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
            }

            #[must_use]
            pub fn distance(&self, other: &PartialRecExpr) -> usize {
                $crate::tree_distance::distance(&self.expr, &other.expr)
            }

            #[pyo3(signature = (name, path, transparent=false))]
            pub fn to_dot(&self, name: String, path: String, transparent: bool) {
                let mut path = std::env::current_dir().unwrap().join(path);
                let dot = $crate::viz::to_dot(&self.expr, &name, transparent);
                path.set_extension("dot");
                std::fs::write(&path, &dot).unwrap();

                let svg = $crate::viz::dot_to_svg(&dot);
                path.set_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }

            #[staticmethod]
            #[expect(clippy::missing_errors_doc)]
            pub fn count_expected_tokens(token_list: Vec<String>) -> PyResult<usize> {
                Ok(partial::count_expected_tokens::<L, _>(&token_list)?)
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn lower_meta_level(&self) -> PyResult<RecExpr> {
                let r = partial::lower_meta_level(self.expr.clone())?;
                Ok(RecExpr(r))
            }
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn operators() -> Vec<String> {
            L::operators().iter().map(|s| s.to_string()).collect()
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn name_to_id(s: String) -> Option<usize> {
            L::operators().iter().position(|o| o == &s)
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn num_symbols() -> usize {
            L::NUM_SYMBOLS
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn eqsat_check(
            start: &RecExpr,
            goal: &RecExpr,
            iter_limit: usize,
        ) -> (usize, String, String) {
            let conf = EqsatConf::builder()
                .root_check(true)
                .iter_limit(iter_limit)
                .build();
            let start_expr = &[&start.0];
            let start_material = StartMaterial::RecExprs(start_expr);
            let rules = <$type as TermRewriteSystem>::full_rules();
            let eqsat_result = Eqsat::new(start_material, &rules)
                .with_conf(conf)
                .with_goals(vec![goal.0.clone()])
                .run();
            let generation = eqsat_result.iterations().len();
            let stop_reason = serde_json::to_string(&eqsat_result.report().stop_reason).unwrap();
            let report_json = serde_json::to_string(&eqsat_result).unwrap();
            (generation, stop_reason, report_json)
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn many_eqsat_check(
            starts: Vec<RecExpr>,
            goal: &RecExpr,
            iter_limit: usize,
        ) -> Vec<(usize, String, String)> {
            starts
                .into_iter()
                .map(|start| eqsat_check(&start, goal, iter_limit))
                .collect()
        }

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<RecExpr>()?;
            module.add_class::<PartialRecExpr>()?;

            module.add_function(pyo3::wrap_pyfunction!(operators, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(name_to_id, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(num_symbols, m)?)?;

            module.add_function(pyo3::wrap_pyfunction!(eqsat_check, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(many_eqsat_check, m)?)?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub(crate) use monomorphize;
