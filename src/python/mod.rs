// pub mod halide;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn eggshell(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pylang works for all langs that implement Display
    // m.add_class::<python:>()?;
    simple::add_mod(m, "simple")?;
    arithmetic::add_mod(m, "arithmetic")?;
    halide::add_mod(m, "halide")?;
    rise::add_mod(m, "rise")?;

    m.add(
        "EggshellException",
        m.py().get_type::<crate::error::EggshellException>(),
    )?;

    Ok(())
}

/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty, $module_name: tt) => {
        use egg::{FromOp, Id, Language, RecExpr};
        use pyo3::prelude::*;
        use pyo3_stub_gen::derive::*;
        use rayon::prelude::*;

        use $crate::eqsat::conf::EqsatConf;
        use $crate::eqsat::{Eqsat, StartMaterial};
        use $crate::error::EggshellError;
        use $crate::features::AsFeatures;
        use $crate::trs::{MetaInfo, TermRewriteSystem, TrsError};

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
            pub fn new(s_expr_str: &str) -> PyResult<Self> {
                let rec_expr = s_expr_str
                    .parse::<egg::RecExpr<L>>()
                    .map_err(|e| EggshellError::from(e))?;
                Ok(Self(rec_expr))
            }

            #[expect(clippy::missing_errors_doc)]
            #[staticmethod]
            pub fn many_new(s_expr_strs: Vec<String>) -> PyResult<Vec<Self>> {
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
            pub fn children_of(&self, node: &PyNode) -> Vec<PyNode> {
                node.0
                    .children()
                    .iter()
                    .map(|c_id| PyNode(self.0[*c_id].to_owned()))
                    .collect()
            }

            #[must_use]
            pub fn arity(&self, position: usize) -> usize {
                self.0.arity(position)
            }

            #[must_use]
            pub fn size(&self) -> usize {
                self.0.size()
            }

            #[must_use]
            pub fn depth(&self) -> usize {
                self.0.depth()
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn count_symbols(
                &self,
                variable_names: Vec<String>,
                ignore_unknown: bool,
            ) -> PyResult<Vec<usize>> {
                let x = self
                    .0
                    .count_symbols(&variable_names, ignore_unknown)
                    .map_err(|e| EggshellError::<L>::from(e))?;
                Ok(x)
            }

            #[expect(clippy::cast_precision_loss, clippy::missing_errors_doc)]
            pub fn featurize_simple<'py>(
                &self,
                py: Python<'py>,
                variable_names: Vec<String>,
                ignore_unknown: bool,
            ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
                let mut features = self
                    .0
                    .count_symbols(&variable_names, ignore_unknown)
                    .map_err(|e| EggshellError::<L>::from(e))?;
                features.push(self.0.size());
                features.push(self.0.depth());
                let rust_vec = features.into_iter().map(|v| v as f64).collect();
                Ok(numpy::PyArray::from_vec(py, rust_vec))
            }

            #[must_use]
            pub fn feature_names_simple(&self) -> Vec<&str> {
                let mut symbol_names = L::operators();
                symbol_names.push("SIZE");
                symbol_names.push("DEPTH");
                symbol_names
            }
        }

        #[gen_stub_pyclass]
        #[pyclass(frozen, module = $module_name,)]
        #[derive(Debug, Clone, PartialEq)]
        pub struct PyNode(L);

        #[gen_stub_pymethods]
        #[pymethods]
        impl PyNode {
            #[expect(clippy::missing_errors_doc)]
            #[new]
            pub fn new(node_name: String, children: Vec<usize>) -> PyResult<Self> {
                let c_ids = children.into_iter().map(|id| Id::from(id)).collect();
                let expr =
                    L::from_op(&node_name, c_ids).map_err(|e| EggshellError::<L>::from(e))?;
                Ok(PyNode(expr))
            }

            #[must_use]
            pub fn name(&self) -> String {
                self.0.to_string()
            }

            #[must_use]
            pub fn is_leaf(&self) -> bool {
                self.0.is_leaf()
            }

            #[must_use]
            pub fn children(&self) -> Vec<usize> {
                self.0.children().iter().map(|c| (*c).into()).collect()
            }

            #[expect(clippy::missing_errors_doc)]
            pub fn features(
                &self,
                variable_names: Vec<String>,
                ignore_unknown: bool,
            ) -> PyResult<Vec<f64>> {
                let f = $crate::features::features(&self.0, &variable_names, ignore_unknown)
                    .map_err(|e| EggshellError::<L>::from(e))?
                    .ok_or_else(|| {
                        EggshellError::<L>::from(TrsError::IgnoredSymbol(self.name()))
                    })?;
                Ok(f)
            }
        }

        #[gen_stub_pyfunction(module = $module_name)]
        #[pyfunction]
        #[expect(clippy::cast_precision_loss, clippy::missing_errors_doc)]
        pub fn many_featurize_simple(
            py: Python<'_>,
            exprs: Vec<PyRecExpr>,
            variable_names: Vec<String>,
            ignore_unknown: bool,
        ) -> PyResult<Bound<'_, numpy::PyArray2<f64>>> {
            let rust_vec = exprs
                .par_iter()
                .map(|expr| {
                    let mut features = expr
                        .0
                        .count_symbols(&variable_names, ignore_unknown)
                        .map_err(|e| EggshellError::<L>::from(e))?;
                    features.push(expr.0.size());
                    features.push(expr.0.depth());
                    Ok(features.into_iter().map(|v| v as f64).collect())
                })
                .collect::<Result<Vec<_>, EggshellError<_>>>()?;

            Ok(numpy::PyArray::from_vec2(py, &rust_vec).unwrap())
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

            module.add_function(pyo3::wrap_pyfunction!(many_featurize_simple, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(eqsat_check, m)?)?;
            module.add_function(pyo3::wrap_pyfunction!(many_eqsat_check, m)?)?;

            m.add_submodule(&module)?;
            Ok(())
        }
    };
}

pub mod simple {
    monomorphize!(crate::trs::Simple, "eggshell.simple");
}

pub mod arithmetic {
    monomorphize!(crate::trs::Arithmetic, "eggshell.arithmetic");
}

pub mod halide {
    monomorphize!(crate::trs::Halide, "eggshell.halide");
}

pub mod rise {
    monomorphize!(crate::trs::Rise, "eggshell.rise");
}

use pyo3_stub_gen::define_stub_info_gatherer;
define_stub_info_gatherer!(stub_info);
