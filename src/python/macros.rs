/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone)]
        pub struct Eqsat($crate::eqsat::Eqsat<$type>);

        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone)]
        pub struct EqsatResult($crate::eqsat::EqsatResult<$type>);

        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone)]
        pub struct Rules(
            Vec<
                egg::Rewrite<
                    <$type as $crate::trs::Trs>::Language,
                    <$type as $crate::trs::Trs>::Analysis,
                >,
            >,
        );

        #[pyo3::pymethods]
        impl Eqsat {
            #[new]
            #[pyo3(signature = (start_terms, **py_kwargs))]
            fn new(
                start_terms: Vec<String>,
                py_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
            ) -> pyo3::PyResult<Self> {
                let start_exprs = start_terms
                    .into_iter()
                    .map(|term| term.parse())
                    .collect::<Result<_, _>>()
                    .map_err(|e| $crate::python::EggError::RecExprParse(e))?;

                let eqsat = $crate::eqsat::Eqsat::new(start_exprs);

                if let Some(bound) = py_kwargs {
                    let iter_limit = pyo3::types::PyDictMethods::get_item(bound, "iter_limit")?
                        .map(|t| pyo3::types::PyAnyMethods::extract(&t))
                        .transpose()?;

                    let node_limit = pyo3::types::PyDictMethods::get_item(bound, "node_limit")?
                        .map(|t| pyo3::types::PyAnyMethods::extract(&t))
                        .transpose()?;

                    let time_limit = pyo3::types::PyDictMethods::get_item(bound, "time_limit")?
                        .map(|t| pyo3::types::PyAnyMethods::extract(&t))
                        .transpose()?;

                    let runner_args =
                        $crate::eqsat::EqsatConf::new(false, iter_limit, node_limit, time_limit);
                    Ok(Self(eqsat.with_conf(runner_args)))
                } else {
                    Ok(Self(eqsat))
                }
            }

            #[allow(clippy::missing_errors_doc)]
            fn run(&self, rules: &Rules) -> pyo3::PyResult<EqsatResult> {
                let r = self.0.run(&rules.0);

                Ok(EqsatResult(r))
            }

            #[getter]
            pub fn runner_args(&self) -> $crate::eqsat::EqsatConf {
                self.0.runner_args().clone()
            }
        }

        #[pyo3::pymethods]
        impl EqsatResult {
            #[allow(clippy::missing_errors_doc)]
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

            #[pyo3(signature = (root, sketch, **py_kwargs))]
            fn sketch_extract(
                &mut self,
                root: usize,
                sketch: $crate::python::PySketch,
                py_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
            ) -> pyo3::PyResult<(usize, $crate::python::PyLang)> {
                let (cost, term) = if let Some(bound) = py_kwargs {
                    if let Some(cost_fn_name) =
                        pyo3::types::PyDictMethods::get_item(bound, "cost_function")?
                    {
                        let cost_fn_name = pyo3::types::PyAnyMethods::extract(&cost_fn_name)?;
                        match cost_fn_name {
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
                                    format!("{cost_fn_name} is not a valid cost function"),
                                ))
                            }
                        }
                    } else {
                        return Err(pyo3::PyErr::new::<pyo3::exceptions::PySyntaxError, _>(
                            "Specified non-existent arguments",
                        ));
                    }
                } else {
                    self.0
                        .sketch_extract(root.into(), $crate::utils::AstSize2, &sketch.try_into()?)
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

            // #[getter]
            // pub fn egraph(
            //     &self,
            // ) -> egg::EGraph<
            //     <$type as $crate::trs::Trs>::Language,
            //     <$type as $crate::trs::Trs>::Analysis,
            // > {
            //     self.0.egraph().clone()
            // }

            // #[allow(clippy::missing_panics_doc)]
            // pub fn sketch_extract<CF>(
            //     &self,
            //     root: Id,
            //     cost_fn: CF,
            //     sketch: &Sketch<R::Language>,
            // ) -> (CF::Cost, RecExpr<R::Language>)
            // where
            //     CF: CostFunction<R::Language>,
            //     CF::Cost: Ord,
            // {
            //     extract::eclass_extract(sketch, cost_fn, &self.egraph, root).unwrap()
            // }
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
