/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::{Language, RecExpr};
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::*;
        use rayon::prelude::*;

        use $crate::eqsat::conf::EqsatConf;
        use $crate::eqsat::{Eqsat, StartMaterial};
        use $crate::meta_lang;
        use $crate::python::data::TreeData;
        use $crate::python::err::EggshellError;
        use $crate::trs::{MetaInfo, TermRewriteSystem};

        type L = <$type as TermRewriteSystem>::Language;
        // type N = <$type as TermRewriteSystem>::Analysis;

        #[gen_stub_pyclass]
        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct PyRecExpr(RecExpr<L>);

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyRecExpr {
            /// Parse from string
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(s_expr_str: &str) -> PyResult<PyRecExpr> {
                let rec_expr = s_expr_str
                    .parse::<egg::RecExpr<L>>()
                    .map_err(|e| EggshellError::from(e))?;
                Ok(PyRecExpr(rec_expr))
            }

            #[expect(clippy::missing_errors_doc)]
            #[staticmethod]
            pub fn batch_new(s_expr_strs: Vec<String>) -> PyResult<Vec<PyRecExpr>> {
                s_expr_strs.par_iter().map(|s| PyRecExpr::new(s)).collect()
            }

            #[must_use]
            fn __str__(&self) -> String {
                self.0.to_string()
            }

            #[must_use]
            pub fn __repr__(&self) -> String {
                format!("{self:?}")
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
            #[must_use]
            pub fn to_data_batch(rec_exprs: Vec<PyRecExpr>) -> Vec<TreeData> {
                rec_exprs.into_par_iter().map(|r| (&r.0).into()).collect()
            }

            #[staticmethod]
            #[expect(clippy::missing_errors_doc)]
            pub fn from_data(tree_data: TreeData) -> PyResult<PyRecExpr> {
                let expr = (&tree_data)
                    .try_into()
                    .map_err(|e| EggshellError::<L>::from(e))?;
                Ok(PyRecExpr(expr))
            }
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[expect(clippy::missing_errors_doc)]
        pub fn partial_parse(token_list: Vec<String>) -> PyResult<TreeData> {
            let r = (&meta_lang::partial_parse::<L, _>(token_list.as_slice())?).into();
            Ok(r)
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[expect(clippy::missing_errors_doc)]
        pub fn lower_meta_level(token_list: Vec<String>) -> PyResult<PyRecExpr> {
            let rec_expr = meta_lang::partial_parse::<L, _>(token_list.as_slice())?;
            let r = meta_lang::lower_meta_level::<L>(&rec_expr)?;
            Ok(PyRecExpr(r))
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
            start: &PyRecExpr,
            goal: &PyRecExpr,
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
            starts: Vec<PyRecExpr>,
            goal: &PyRecExpr,
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
            module.add_class::<PyRecExpr>()?;

            module.add_function(pyo3::wrap_pyfunction!(partial_parse, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(lower_meta_level, m)?)?;

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
