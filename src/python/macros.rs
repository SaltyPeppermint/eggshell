/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        // /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        // #[pyo3::pyclass]
        // #[derive(Debug, Clone)]
        // pub struct Eqsat($crate::eqsat::Eqsat<$type>);

        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone)]
        pub struct EqsatResult($crate::eqsat::EqsatResult<$type>);

        #[pyo3::pyfunction]
        pub fn run_eqsat(
            start_terms: Vec<$crate::python::PyLang>,
            ruleset_name: String,
            conf: Option<$crate::eqsat::utils::EqsatConf>,
        ) -> pyo3::PyResult<EqsatResult> {
            let start_exprs = start_terms
                .into_iter()
                .map(|term| (&term).try_into())
                // Vec<egg::RecExpr<<$type as $crate::trs::Trs>::Language>>
                .collect::<Result<_, _>>()
                .map_err(|e| {
                    $crate::python::EggError::FromOp::<
                        <<$type as $crate::trs::Trs>::Language as egg::FromOp>::Error,
                    >(e)
                })?;

            let eqsat = if let Some(c) = conf {
                $crate::eqsat::Eqsat::new(start_exprs).with_conf(c)
            } else {
                $crate::eqsat::Eqsat::new(start_exprs)
            };

            let ruleset = <$type as $crate::trs::Trs>::Rulesets::try_from(ruleset_name)?;
            let rules = <$type as $crate::trs::Trs>::rules(&ruleset);
            let result = eqsat.run(&rules);
            Ok(EqsatResult(result))
        }

        #[pyo3::pymethods]
        impl EqsatResult {
            #[pyo3(signature = (root, cost_fn="ast_size"))]
            fn classic_extract(
                &mut self,
                root: usize,
                cost_fn: &str,
            ) -> pyo3::PyResult<(usize, $crate::python::PyLang)> {
                let (cost, term) = {
                    match cost_fn {
                        "ast_size" => self.0.classic_extract(root.into(), $crate::utils::AstSize2),
                        "ast_depth" => self
                            .0
                            .classic_extract(root.into(), $crate::utils::AstDepth2),
                        _ => {
                            return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("{cost_fn} is not a valid cost function"),
                            ))
                        }
                    }
                };
                Ok((cost, (&term).into()))
            }

            #[pyo3(signature = (root, sketch,cost_fn="ast_size"))]
            fn sketch_extract(
                &mut self,
                root: usize,
                sketch: $crate::python::PySketch,
                cost_fn: &str,
            ) -> pyo3::PyResult<(usize, $crate::python::PyLang)> {
                let (cost, term) = match cost_fn {
                    "ast_size" => self.0.sketch_extract(
                        root.into(),
                        $crate::utils::AstSize2,
                        &sketch.try_into()?,
                    ),
                    "ast_depth" => self.0.sketch_extract(
                        root.into(),
                        $crate::utils::AstDepth2,
                        &sketch.try_into()?,
                    ),
                    _ => {
                        return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("{cost_fn} is not a valid cost function"),
                        ))
                    }
                };
                Ok((cost, (&term).into()))
            }

            #[getter]
            pub fn roots(&self) -> Vec<usize> {
                self.0.roots().iter().map(|id| (*id).into()).collect()
            }

            pub fn report(&self) -> String {
                self.0.report().to_string()
            }
        }
    };
}

/// Macro to generate the necessary implementations so pyo3 doesnt freak out about
/// self-referential types using boxes
macro_rules! pyboxable {
    ($type: ty) => {
        impl<'source> pyo3::FromPyObject<'source> for std::boxed::Box<$type> {
            fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
                ob.extract::<$type>().map(Box::new)
            }
        }
        impl pyo3::IntoPy<pyo3::PyObject> for std::boxed::Box<$type> {
            fn into_py(self, py: pyo3::Python<'_>) -> pyo3::PyObject {
                (*self).into_py(py)
            }
        }
    };
}

pub(crate) use monomorphize;
pub(crate) use pyboxable;
