/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::Language;
        use egg::RecExpr as EggRecExpr;
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::*;

        use $crate::eqsat::Eqsat;
        use $crate::eqsat::conf::EqsatConf;
        use $crate::meta_lang::partial;
        use $crate::meta_lang::probabilistic;
        use $crate::meta_lang::probabilistic::FirstErrorDistance;
        use $crate::meta_lang::{PartialLang, ProbabilisticLang};
        use $crate::python::err::EggshellError;
        use $crate::rewrite_system::{LangExtras, RewriteSystem};
        use $crate::tree_data::TreeData;

        type L = <$type as RewriteSystem>::Language;

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
                let rec_expr: egg::RecExpr<L> = s_expr_str
                    .parse()
                    .map_err(|e: egg::RecExprParseError<_>| EggshellError::<L>::from(e))?;
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

            #[pyo3(signature = (name, path, marked_ids=None, transparent=false))]
            pub fn to_dot(
                &self,
                name: String,
                path: String,
                marked_ids: Option<Vec<usize>>,
                transparent: bool,
            ) {
                let dot = $crate::viz::to_dot(
                    &self.0,
                    &name,
                    &(marked_ids
                        .unwrap_or_default()
                        .into_iter()
                        .map(|id| egg::Id::from(id))
                        .collect()),
                    transparent,
                );
                let svg = $crate::viz::dot_to_svg(&dot);
                let path = std::env::current_dir()
                    .unwrap()
                    .join(path)
                    .with_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }

            #[must_use]
            pub fn tree_distance(&self, other: &RecExpr) -> usize {
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
        pub struct GeneratedRecExpr {
            expr: EggRecExpr<PartialLang<ProbabilisticLang<L>>>,
            #[pyo3(get)]
            used_tokens: usize,
        }

        #[gen_stub_pymethods]
        #[pymethods]
        impl GeneratedRecExpr {
            #[expect(clippy::missing_errors_doc)]
            #[new]
            #[pyo3(signature = (token_list, token_probs=None))]
            pub fn new(
                token_list: Vec<String>,
                token_probs: Option<Vec<f64>>,
            ) -> PyResult<GeneratedRecExpr> {
                let (partial_node, used_tokens) = partial::partial_parse::<L, _>(
                    token_list.as_slice(),
                    token_probs.as_ref().map(|v| &**v),
                )
                .map_err(|e| EggshellError::from(e))?;

                Ok(GeneratedRecExpr {
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
            pub fn tree_distance(&self, other: &GeneratedRecExpr) -> usize {
                $crate::tree_distance::distance(&self.expr, &other.expr)
            }

            #[pyo3(signature = (name, path, marked_ids=None, transparent=false))]
            pub fn to_dot(
                &self,
                name: String,
                path: String,
                marked_ids: Option<Vec<usize>>,
                transparent: bool,
            ) {
                let dot = $crate::viz::to_dot(
                    &self.expr,
                    &name,
                    &(marked_ids
                        .unwrap_or_default()
                        .into_iter()
                        .map(|id| egg::Id::from(id))
                        .collect()),
                    transparent,
                );
                let svg = $crate::viz::dot_to_svg(&dot);
                let path = std::env::current_dir()
                    .unwrap()
                    .join(path)
                    .with_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }

            #[staticmethod]
            #[expect(clippy::missing_errors_doc)]
            pub fn count_expected_tokens(token_list: Vec<String>) -> PyResult<usize> {
                Ok(partial::count_expected_tokens::<L, _>(&token_list)
                    .map_err(|e| EggshellError::from(e))?)
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn lower(&self) -> PyResult<RecExpr> {
                let r = PartialLang::lower(&self.expr).map_err(|e| EggshellError::from(e))?;
                let r = ProbabilisticLang::lower(&r);
                Ok(RecExpr(r))
            }
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        pub fn first_miss_distance(
            ground_truth: &RecExpr,
            generated: &GeneratedRecExpr,
        ) -> PyResult<FirstErrorDistance> {
            let inner = PartialLang::lower(&generated.expr).map_err(|e| EggshellError::from(e))?;
            Ok(probabilistic::compare(&ground_truth.0, &inner))
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
        #[pyo3(signature = (start, goal, iter_limit=None, node_limit=None, guides=Vec::new()))]
        #[must_use]
        pub fn eqsat_check(
            start: &RecExpr,
            goal: &RecExpr,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            guides: Vec<RecExpr>,
        ) -> (usize, String, String) {
            let conf = EqsatConf::builder()
                .maybe_iter_limit(iter_limit)
                .maybe_node_limit(node_limit)
                .build();
            let rules = <$type as RewriteSystem>::full_rules();
            let guides = guides.into_iter().map(|r| r.0).collect::<Vec<_>>();
            let eqsat_result = Eqsat::new((&start.0).into(), &rules)
                .with_conf(conf)
                .with_goal(goal.0.clone())
                .with_guides(&guides)
                .run();
            let generation = eqsat_result.iterations().len();
            let stop_reason = serde_json::to_string(&eqsat_result.report().stop_reason).unwrap();
            let report_json = serde_json::to_string(&eqsat_result).unwrap();
            (generation, stop_reason, report_json)
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[must_use]
        #[pyo3(signature = (starts, goals, iter_limit=None, node_limit=None, guides=Vec::new()))]
        pub fn many_eqsat_check(
            starts: Vec<RecExpr>,
            goals: Vec<RecExpr>,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            guides: Vec<Vec<RecExpr>>,
        ) -> Vec<(usize, String, String)> {
            starts
                .iter()
                .zip(goals.iter())
                .enumerate()
                .map(|(idx, (start, goal))| {
                    eqsat_check(
                        &start,
                        &goal,
                        iter_limit,
                        node_limit,
                        guides
                            .get(idx)
                            .map(|v| v.to_owned())
                            .unwrap_or_else(Vec::new),
                    )
                })
                .collect()
        }

        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<RecExpr>()?;
            module.add_class::<GeneratedRecExpr>()?;

            module.add_function(pyo3::wrap_pyfunction!(first_miss_distance, m)?)?;

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
