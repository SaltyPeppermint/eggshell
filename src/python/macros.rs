/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone, serde::Serialize)]
        pub struct NewEqsat($crate::eqsat::Eqsat<$type, $crate::eqsat::New>);

        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass]
        #[derive(Debug, Clone, serde::Serialize)]
        pub struct FinishedEqsat($crate::eqsat::Eqsat<$type, $crate::eqsat::Finished>);

        #[pyo3::pymethods]
        impl NewEqsat {
            #[new]
            #[pyo3(signature = (index, **py_kwargs))]
            fn new(
                index: usize,
                py_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
            ) -> pyo3::PyResult<Self> {
                let eqsat = $crate::eqsat::Eqsat::new(index);
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
                        $crate::eqsat::utils::RunnerArgs::new(iter_limit, node_limit, time_limit);
                    Ok(Self(eqsat.with_runner_args(runner_args)))
                } else {
                    Ok(Self(eqsat))
                }
            }

            #[allow(clippy::missing_errors_doc)]
            fn run(&mut self, start_term: &str) -> pyo3::PyResult<FinishedEqsat> {
                let start_expr = start_term
                    .parse()
                    .map_err(|e| $crate::errors::EggError::RecExprParse(e))?;
                let rules = <$type as $crate::trs::Trs>::rules(
                    &<$type as $crate::trs::Trs>::maximum_ruleset(),
                );
                let r = self.0.run(&start_expr, &rules);

                Ok(FinishedEqsat(r))
            }

            pub fn get_runner_args(&self) -> $crate::eqsat::utils::RunnerArgs {
                self.0.runner_args().clone()
            }
        }

        #[pyo3::pymethods]
        impl FinishedEqsat {
            #[allow(clippy::missing_errors_doc)]
            #[pyo3(signature = (**py_kwargs))]
            fn extract(
                &mut self,
                py_kwargs: Option<&pyo3::Bound<'_, pyo3::types::PyDict>>,
            ) -> pyo3::PyResult<Vec<(usize, $crate::python::PyLang)>> {
                let extracted = if let Some(bound) = py_kwargs {
                    if let Some(cost_fn_name) =
                        pyo3::types::PyDictMethods::get_item(bound, "cost_function")?
                    {
                        let cost_fn_name = pyo3::types::PyAnyMethods::extract(&cost_fn_name)?;
                        match cost_fn_name {
                            "ast_size" => self.0.classic_extract($crate::utils::AstSize2),
                            "ast_depth" => self.0.classic_extract($crate::utils::AstDepth2),
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
                    self.0.classic_extract($crate::utils::AstSize2)
                };
                Ok(extracted
                    .iter()
                    .map(|(cost, term)| (*cost, term.into()))
                    .collect::<Vec<_>>())
            }

            pub fn get_runner_args(&self) -> $crate::eqsat::utils::RunnerArgs {
                self.0.runner_args().clone()
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
