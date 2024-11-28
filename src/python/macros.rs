/// Macro to generate a manuaol monomorphization via a wrapper cause
/// pyo3 can't handle generics.
macro_rules! monomorphize {
    ($type: ty) => {
        pub(crate) fn add_mod(
            m: &pyo3::Bound<'_, pyo3::prelude::PyModule>,
            module_name: &str,
        ) -> pyo3::PyResult<()> {
            use pyo3::prelude::PyModuleMethods;

            let bound = pyo3::prelude::PyModule::new_bound(m.py(), module_name)?;
            bound.add_function(pyo3::wrap_pyfunction!(symbols, m)?)?;
            bound.add_function(pyo3::wrap_pyfunction!(symbol_table, m)?)?;
            bound.add_function(pyo3::wrap_pyfunction!(run_eqsat, m)?)?;
            // bound.add_function(pyo3::wrap_pyfunction!(syntaxcheck_expr, m)?)?;
            // bound.add_function(pyo3::wrap_pyfunction!(typecheck_expr, m)?)?;
            // bound.add_function(pyo3::wrap_pyfunction!(syntaxcheck_sketch, m)?)?;
            // bound.add_function(pyo3::wrap_pyfunction!(typecheck_sketch, m)?)?;
            bound.add_class::<EqsatResult>()?;
            m.add_submodule(&bound)?;
            Ok(())
        }

        type Lang = <$type as $crate::trs::TermRewriteSystem>::Language;

        /// Gets the symbols inherent to the language
        #[pyo3::pyfunction]
        pub fn symbols(variables: usize) -> Vec<(String, usize)> {
            <Lang as $crate::trs::TrsLang>::symbols(variables)
        }

        /// Gets the symbols inherent to the language
        #[pyo3::pyfunction]
        pub fn symbol_table(variables: usize) -> $crate::python::symbols::SymbolTable {
            let lut = <Lang as $crate::trs::TrsLang>::symbol_lut(variables);
            $crate::python::symbols::SymbolTable::new(lut)
        }

        // /// Check if a expression has the correct syntax
        // #[pyo3::pyfunction]
        // pub fn syntaxcheck_expression(expression: &$crate::python::PyLang) -> bool {
        //     let expr: Result<egg::RecExpr<Lang>, _> = (&expression.0).try_into();
        //     expr.is_ok()
        // }

        // /// Check if a expression typechecks
        // #[pyo3::pyfunction]
        // pub fn typecheck_expression(expression: &$crate::python::PyLang) -> pyo3::PyResult<bool> {
        //     let expr: egg::RecExpr<Lang> = (&expression.0)
        //         .try_into()
        //         .map_err(|e| $crate::python::EggError::FromOp::<<Lang as egg::FromOp>::Error>(e))?;
        //     Ok($crate::typing::typecheck_expr(&expr).is_ok())
        // }

        // /// Check if a partial sketch has the correct syntax
        // #[pyo3::pyfunction]
        // pub fn syntaxcheck_sketch(expression: &$crate::python::PySketch) -> bool {
        //     let sketch: Result<egg::RecExpr<$crate::sketch::PartialSketchNode<Lang>>, _> =
        //         (&expression.0).try_into();
        //     sketch.is_ok()
        // }

        // /// Check if a partial sketch typechecks
        // #[pyo3::pyfunction]
        // pub fn typecheck_sketch(expression: &$crate::python::PySketch) -> pyo3::PyResult<bool> {
        //     let sketch: egg::RecExpr<$crate::sketch::PartialSketchNode<Lang>> =
        //         (&expression.0).try_into()?;
        //     Ok($crate::typing::typecheck_expr(&sketch).is_ok())
        // }

        /// Run an eqsat on the expr
        #[pyo3::pyfunction]
        #[pyo3(signature = (start_asts, conf=None))]
        pub fn run_eqsat(
            start_asts: Vec<$crate::python::PyAst>,
            conf: Option<$crate::eqsat::EqsatConf>,
        ) -> pyo3::PyResult<EqsatResult> {
            let start_exprs = start_asts
                .into_iter()
                .map(|start_ast| (&start_ast.0).try_into())
                .collect::<Result<_, _>>()?;
            let eqsat = if let Some(c) = conf {
                $crate::eqsat::Eqsat::new($crate::eqsat::StartMaterial::RecExprs(start_exprs))
                    .with_conf(c)
            } else {
                $crate::eqsat::Eqsat::new($crate::eqsat::StartMaterial::RecExprs(start_exprs))
            };

            let rules = <$type as $crate::trs::TermRewriteSystem>::full_rules();
            let result = eqsat.run(&rules);
            Ok(EqsatResult(result))
        }

        /// Manual wrapper (or monomorphization) of [`Eqsat`] to work around Pyo3 limitations
        #[pyo3::pyclass(frozen)]
        #[derive(Debug, Clone)]
        pub struct EqsatResult(
            $crate::eqsat::EqsatResult<
                <$type as $crate::trs::TermRewriteSystem>::Language,
                <$type as $crate::trs::TermRewriteSystem>::Analysis,
            >,
        );

        #[pyo3::pymethods]
        impl EqsatResult {
            #[pyo3(signature = (root, cost_fn="ast_size"))]
            fn classic_extract(
                &self,
                root: usize,
                cost_fn: &str,
            ) -> pyo3::PyResult<(usize, $crate::python::PyAst)> {
                let (cost, expr) = {
                    match cost_fn {
                        "ast_size" => self.0.classic_extract(root.into(), egg::AstSize),
                        "ast_depth" => self.0.classic_extract(root.into(), egg::AstDepth),
                        _ => {
                            return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("{cost_fn} is not a valid cost function"),
                            ))
                        }
                    }
                };
                Ok((cost, (&expr).into()))
            }

            #[pyo3(signature = (root, sketch,cost_fn="ast_size"))]
            fn sketch_extract(
                &self,
                root: usize,
                sketch: $crate::python::PyAst,
                cost_fn: &str,
            ) -> pyo3::PyResult<(usize, $crate::python::PyAst)> {
                let sketch = &(&sketch.0).try_into()?;
                let (cost, expr) = match cost_fn {
                    "ast_size" => self.0.sketch_extract(root.into(), egg::AstSize, sketch),
                    "ast_depth" => self.0.sketch_extract(root.into(), egg::AstDepth, sketch),
                    _ => {
                        return Err(pyo3::PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("{cost_fn} is not a valid cost function"),
                        ))
                    }
                };
                Ok((cost, (&expr).into()))
            }

            fn table_extract(
                &self,
                root: usize,
                table: hashbrown::HashMap<(usize, usize), f64>,
            ) -> (f64, $crate::python::PyAst) {
                let t = table
                    .into_iter()
                    .map(|(k, v)| ((k.0.into(), k.1), v))
                    .collect();
                let (cost, expr) = self.0.table_extract(root.into(), t);
                (cost, (&expr).into())
            }

            fn flat_egraph(&self) -> $crate::python::flat::FlatEGraph {
                (&self.0).into()
            }

            #[getter]
            pub fn roots(&self) -> Vec<usize> {
                self.0.roots().iter().map(|id| (*id).into()).collect()
            }

            #[getter]
            pub fn report(&self) -> String {
                self.0.report().to_string()
            }
        }
    };
}

/// Macro to generate the necessary implementations so pyo3 doesnt freak out about
/// self-referential types using boxes
// macro_rules! pyboxable {
//     ($type: ty) => {
//         impl pyo3::FromPyObject<'_> for std::boxed::Box<$type> {
//             fn extract_bound(ob: &Bound<'_, PyAny>) -> pyo3::PyResult<Self> {
//                 ob.extract::<$type>().map(Box::new)
//             }
//         }
//         impl pyo3::IntoPy<pyo3::PyObject> for std::boxed::Box<$type> {
//             fn into_py(self, py: pyo3::Python<'_>) -> pyo3::PyObject {
//                 (*self).into_py(py)
//             }
//         }
//     };
// }
pub(crate) use monomorphize;
// pub(crate) use pyboxable;
