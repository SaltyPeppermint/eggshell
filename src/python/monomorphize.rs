/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::RecExpr as EggRecExpr;
        use pyo3::prelude::*;

        use $crate::eqsat::{self, EqsatConf};
        use $crate::meta_lang::Sketch;
        use $crate::python::err::EggshellError;
        use $crate::rewrite_system::{LangExtras, RewriteSystem};

        type L = <$type as RewriteSystem>::Language;

        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct RecExpr(EggRecExpr<L>);

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
                let path = std::path::PathBuf::from(path)
                    .with_extension("svg");
                std::fs::write(&path, &svg).unwrap();
            }

            #[must_use]
            pub fn tree_distance(&self, other: &RecExpr) -> usize {
                $crate::tree_distance::distance(&self.0, &other.0)
            }
        }

        #[pyclass(frozen, module = $module_name)]
        #[derive(Debug, Clone, PartialEq)]
        /// Wrapper type for Python
        pub struct Guide(Sketch<L>);

        #[pymethods]
        impl Guide {
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(
                s_expr_str: String,
            ) -> PyResult<Guide> {
                let sketch: Sketch<L> = s_expr_str
                    .parse()
                    .map_err(|e: egg::RecExprParseError<_>| EggshellError::<L>::from(e))?;
                Ok(Guide(sketch))
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
            pub fn tree_distance(&self, other: &Guide) -> usize {
                $crate::tree_distance::distance(&self.0, &other.0)
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
        }

        #[pyfunction]
        #[must_use]
        pub fn operators() -> Vec<String> {
            L::operators().iter().map(|s| s.to_string()).collect()
        }

        #[pyfunction]
        #[must_use]
        pub fn name_to_id(s: String) -> Option<usize> {
            L::operators().iter().position(|o| o == &s)
        }

        #[pyfunction]
        #[must_use]
        pub fn num_symbols() -> usize {
            L::NUM_SYMBOLS
        }

        #[pyfunction]
        #[pyo3(signature = (start, goal, iter_limit=None, node_limit=None, time_limit=None, guide=None))]
        #[must_use]
        pub fn eqsat_guide_check(
            start: &RecExpr,
            goal: &RecExpr,
            iter_limit: Option<usize>,
            node_limit: Option<usize>,
            time_limit: Option<f64>,
            guide: Option<Guide>,
        ) -> (String, bool) {
            let conf = EqsatConf::builder()
                .maybe_iter_limit(iter_limit)
                .maybe_node_limit(node_limit)
                .maybe_time_limit(time_limit.map(std::time::Duration::from_secs_f64))
                .build();

            let eqsat_result = eqsat::eqsat(
                conf,
                (&start.0).into(),
                &<$type as RewriteSystem>::full_rules(),
                Some(goal.0.clone()),
                guide.map(|g|g.0),
                egg::SimpleScheduler,
            );

            let report_json = serde_json::to_string(&eqsat_result).unwrap();
            let guide_used = if let egg::StopReason::Other(report_string) = &eqsat_result.report().stop_reason  {
                report_string.contains("Guide found")
            } else {
                false
            };
            (report_json, guide_used)
        }



        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let module = pyo3::prelude::PyModule::new(m.py(), module_name)?;
            module.add_class::<RecExpr>()?;
            module.add_class::<Guide>()?;

            module.add_function(pyo3::wrap_pyfunction!(operators, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(name_to_id, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(num_symbols, m)?)?;

            module.add_function(pyo3::wrap_pyfunction!(eqsat_guide_check, m)?)?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub(crate) use monomorphize;
