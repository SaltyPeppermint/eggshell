//! Expression types in Rise.

use std::fmt::{self, Display};
use std::str::FromStr;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use symbolic_expressions::{IntoSexp, Sexp};

use super::ParseError;
use super::address::{Address, parse_address};
use super::label::RiseLabel;
use super::nat::{Nat, parse_nat};
use super::primitive::Primitive;
use super::types::{DataType, Type, parse_data_type, parse_type};
use crate::distance::tree::TreeNode;

// ============================================================================
// Literal data values
// ============================================================================

/// Literal data values in Rise.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LiteralData {
    /// Boolean: true/false
    Bool(bool),
    /// Integer: <n>i
    Int(i32),
    /// Float: <n>f or <n>.0
    Float(OrderedFloat<f32>),
    /// Double: <n>d
    Double(OrderedFloat<f64>),
    /// Nat: <n>n (in expression position)
    Nat(i64),
}

impl Display for LiteralData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiteralData::Bool(true) => write!(f, "true"),
            LiteralData::Bool(false) => write!(f, "false"),
            LiteralData::Int(n) => write!(f, "{n}i"),
            LiteralData::Float(n) => write!(f, "{}f", n.0),
            LiteralData::Double(n) => write!(f, "{}d", n.0),
            LiteralData::Nat(n) => write!(f, "{n}n"),
        }
    }
}

impl IntoSexp for LiteralData {
    fn into_sexp(&self) -> Sexp {
        Sexp::String(self.to_string())
    }
}

impl LiteralData {
    /// Convert this literal to a `RiseLabel`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            LiteralData::Bool(b) => RiseLabel::BoolLit(*b),
            LiteralData::Int(n) => RiseLabel::IntLit(*n),
            LiteralData::Float(f) => RiseLabel::FloatLit(*f),
            LiteralData::Double(d) => RiseLabel::DoubleLit(*d),
            LiteralData::Nat(n) => RiseLabel::NatLit(*n),
        }
    }
}

// ============================================================================
// Expression nodes (without type annotations)
// ============================================================================

/// Expression nodes in Rise (the structure without type info).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    /// Variable: $e<index>
    Var(usize),
    /// Application: (app f e)
    App(Box<Expr>, Box<Expr>),
    /// Lambda: (lam body)
    Lambda(Box<Expr>),
    /// Nat application: (natApp f n)
    NatApp(Box<Expr>, Nat),
    /// Nat lambda: (natLam body)
    NatLambda(Box<Expr>),
    /// Data type application: (dataApp f dt)
    DataApp(Box<Expr>, DataType),
    /// Data type lambda: (dataLam body)
    DataLambda(Box<Expr>),
    /// Address application: (addrApp f addr)
    AddrApp(Box<Expr>, Address),
    /// Address lambda: (addrLam body)
    AddrLambda(Box<Expr>),
    /// Nat To Nat lambda: (natNatLam body)
    NatNatLambda(Box<Expr>),
    /// Literal value
    Literal(LiteralData),
    /// Nat literal in expression position: (nat n)
    NatLiteral(Nat),
    /// Index literal: (idxL i n)
    IndexLiteral(Nat, Nat),
    /// Primitive operation
    Primitive(Primitive),
}

impl ExprNode {
    /// Convert this expression node to a `RiseLabel`.
    #[must_use]
    pub fn to_label(&self) -> RiseLabel {
        match self {
            ExprNode::Var(i) => RiseLabel::Var(*i),
            ExprNode::App(..) => RiseLabel::App,
            ExprNode::Lambda(..) => RiseLabel::Lambda,
            ExprNode::NatApp(..) => RiseLabel::NatApp,
            ExprNode::NatLambda(..) => RiseLabel::NatLambda,
            ExprNode::DataApp(..) => RiseLabel::DataApp,
            ExprNode::DataLambda(..) => RiseLabel::DataLambda,
            ExprNode::AddrApp(..) => RiseLabel::AddrApp,
            ExprNode::AddrLambda(..) => RiseLabel::AddrLambda,
            ExprNode::NatNatLambda(..) => RiseLabel::NatNatLambda,
            ExprNode::Literal(lit) => lit.to_label(),
            ExprNode::NatLiteral(n) => n.to_label(),
            ExprNode::IndexLiteral(..) => RiseLabel::IndexLiteral,
            ExprNode::Primitive(p) => RiseLabel::Primitive(p.clone()),
        }
    }
}

impl Display for ExprNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprNode::Var(i) => write!(f, "$e{i}"),
            ExprNode::App(func, arg) => write!(f, "(app {func} {arg})"),
            ExprNode::Lambda(body) => write!(f, "(lam {body})"),
            ExprNode::NatApp(func, n) => write!(f, "(natApp {func} {n})"),
            ExprNode::NatLambda(body) => write!(f, "(natLam {body})"),
            ExprNode::DataApp(func, dt) => write!(f, "(dataApp {func} {dt})"),
            ExprNode::DataLambda(body) => write!(f, "(dataLam {body})"),
            ExprNode::AddrApp(func, addr) => write!(f, "(addrApp {func} {addr})"),
            ExprNode::AddrLambda(body) => write!(f, "(addrLam {body})"),
            ExprNode::NatNatLambda(body) => write!(f, "(natNatLam {body})"),
            ExprNode::Literal(lit) => write!(f, "{lit}"),
            ExprNode::NatLiteral(n) => write!(f, "{n}"),
            ExprNode::IndexLiteral(i, n) => write!(f, "(idxL {i} {n})"),
            ExprNode::Primitive(p) => write!(f, "{p}"),
        }
    }
}

// ============================================================================
// Typed expressions
// ============================================================================

/// A Rise expression with optional type annotation.
///
/// When `ty` is `Some`, the expression is represented as `(typeOf expr type)`.
/// When `ty` is `None`, the expression is represented without type wrapper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expr {
    /// The expression node
    pub node: ExprNode,
    /// Optional type annotation
    pub ty: Option<Type>,
}

impl Expr {
    /// Create a new expression with a type.
    #[must_use]
    pub fn typed(node: ExprNode, ty: Type) -> Self {
        Expr { node, ty: Some(ty) }
    }

    /// Create a new expression without a type.
    #[must_use]
    pub fn untyped(node: ExprNode) -> Self {
        Expr { node, ty: None }
    }

    /// Create a variable expression.
    #[must_use]
    pub fn var_node(index: usize) -> Self {
        Expr::untyped(ExprNode::Var(index))
    }

    /// Create an application expression.
    #[must_use]
    pub fn app_node(f: Expr, e: Expr) -> Self {
        Expr::untyped(ExprNode::App(Box::new(f), Box::new(e)))
    }

    /// Create a lambda expression.
    #[must_use]
    pub fn lambda_node(body: Expr) -> Self {
        Expr::untyped(ExprNode::Lambda(Box::new(body)))
    }

    /// Create a primitive expression.
    #[must_use]
    pub fn prim_node(p: Primitive) -> Self {
        Expr::untyped(ExprNode::Primitive(p))
    }

    /// Convert this expression to a `TreeNode<RiseLabel>` for tree edit distance computation.
    #[must_use]
    pub fn to_typed_tree(&self) -> TreeNode<RiseLabel> {
        self.to_tree(true)
    }

    /// Convert this expression to a `TreeNode<RiseLabel>` without type annotations.
    #[must_use]
    pub fn to_untyped_tree(&self) -> TreeNode<RiseLabel> {
        self.to_tree(false)
    }

    /// Convert this expression to a `TreeNode<RiseLabel>`
    #[must_use]
    pub fn to_tree(&self, with_types: bool) -> TreeNode<RiseLabel> {
        let node_tree = self.node_to_tree(with_types);

        if let Some(ty) = &self.ty
            && with_types
        {
            TreeNode::new(RiseLabel::TypeOf, vec![node_tree, ty.to_tree()])
        } else {
            node_tree
        }
    }

    fn node_to_tree(&self, include_types: bool) -> TreeNode<RiseLabel> {
        let label = self.node.to_label();
        match &self.node {
            ExprNode::Var(_) | ExprNode::Literal(_) | ExprNode::Primitive(_) => {
                TreeNode::leaf(label)
            }
            ExprNode::NatLiteral(n) => n.to_tree(),
            ExprNode::App(f, e) => TreeNode::new(
                label,
                vec![f.to_tree(include_types), e.to_tree(include_types)],
            ),
            ExprNode::Lambda(body)
            | ExprNode::NatLambda(body)
            | ExprNode::DataLambda(body)
            | ExprNode::AddrLambda(body)
            | ExprNode::NatNatLambda(body) => {
                TreeNode::new(label, vec![body.to_tree(include_types)])
            }
            ExprNode::NatApp(f, n) => {
                TreeNode::new(label, vec![f.to_tree(include_types), n.to_tree()])
            }
            ExprNode::DataApp(f, dt) => {
                TreeNode::new(label, vec![f.to_tree(include_types), dt.to_tree()])
            }
            ExprNode::AddrApp(f, addr) => {
                TreeNode::new(label, vec![f.to_tree(include_types), addr.to_tree()])
            }
            ExprNode::IndexLiteral(i, n) => TreeNode::new(label, vec![i.to_tree(), n.to_tree()]),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.ty {
            Some(ty) => write!(f, "(typeOf {} {})", self.node, ty),
            None => write!(f, "{}", self.node),
        }
    }
}

impl IntoSexp for Expr {
    fn into_sexp(&self) -> Sexp {
        let node_sexp = match &self.node {
            ExprNode::Var(i) => Sexp::String(format!("$e{i}")),
            ExprNode::App(func, arg) => Sexp::List(vec![
                Sexp::String("app".to_owned()),
                func.into_sexp(),
                arg.into_sexp(),
            ]),
            ExprNode::Lambda(body) => {
                Sexp::List(vec![Sexp::String("lam".to_owned()), body.into_sexp()])
            }
            ExprNode::NatApp(func, n) => Sexp::List(vec![
                Sexp::String("natApp".to_owned()),
                func.into_sexp(),
                n.into_sexp(),
            ]),
            ExprNode::NatLambda(body) => {
                Sexp::List(vec![Sexp::String("natLam".to_owned()), body.into_sexp()])
            }
            ExprNode::DataApp(func, dt) => Sexp::List(vec![
                Sexp::String("dataApp".to_owned()),
                func.into_sexp(),
                dt.into_sexp(),
            ]),
            ExprNode::DataLambda(body) => {
                Sexp::List(vec![Sexp::String("dataLam".to_owned()), body.into_sexp()])
            }
            ExprNode::AddrApp(func, addr) => Sexp::List(vec![
                Sexp::String("addrApp".to_owned()),
                func.into_sexp(),
                addr.into_sexp(),
            ]),
            ExprNode::AddrLambda(body) => {
                Sexp::List(vec![Sexp::String("addrLam".to_owned()), body.into_sexp()])
            }
            ExprNode::NatNatLambda(body) => {
                Sexp::List(vec![Sexp::String("natNatLam".to_owned()), body.into_sexp()])
            }
            ExprNode::Literal(lit) => lit.into_sexp(),
            ExprNode::NatLiteral(n) => n.into_sexp(),
            ExprNode::IndexLiteral(i, n) => Sexp::List(vec![
                Sexp::String("idxL".to_owned()),
                i.into_sexp(),
                n.into_sexp(),
            ]),
            ExprNode::Primitive(p) => p.into_sexp(),
        };

        match &self.ty {
            Some(ty) => Sexp::List(vec![
                Sexp::String("typeOf".to_owned()),
                node_sexp,
                ty.into_sexp(),
            ]),
            None => node_sexp,
        }
    }
}

impl FromStr for Expr {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let sexp = symbolic_expressions::parser::parse_str(s)?;
        parse_expr(&sexp)
    }
}

/// Parse an expression from an S-expression.
pub fn parse_expr(sexp: &Sexp) -> Result<Expr, ParseError> {
    match sexp {
        Sexp::String(s) => {
            let node = parse_expr_atom(s)?;
            Ok(Expr::untyped(node))
        }
        Sexp::List(items) => {
            let head = items
                .first()
                .and_then(|s| match s {
                    Sexp::String(s_inner) => Some(s_inner.as_str()),
                    _ => None,
                })
                .ok_or_else(|| ParseError::Expr("expected expression head".to_owned()))?;

            match head {
                "typeOf" if items.len() == 3 => {
                    let inner = parse_expr(&items[1])?;
                    let ty = parse_type(&items[2])?;
                    Ok(Expr {
                        node: inner.node,
                        ty: Some(ty),
                    })
                }
                "app" if items.len() == 3 => {
                    let f = parse_expr(&items[1])?;
                    let e = parse_expr(&items[2])?;
                    Ok(Expr::untyped(ExprNode::App(Box::new(f), Box::new(e))))
                }
                "lam" if items.len() == 2 => {
                    let body = parse_expr(&items[1])?;
                    Ok(Expr::untyped(ExprNode::Lambda(Box::new(body))))
                }
                "natApp" if items.len() == 3 => {
                    let f = parse_expr(&items[1])?;
                    let n = parse_nat(&items[2])?;
                    Ok(Expr::untyped(ExprNode::NatApp(Box::new(f), n)))
                }
                "natLam" if items.len() == 2 => {
                    let body = parse_expr(&items[1])?;
                    Ok(Expr::untyped(ExprNode::NatLambda(Box::new(body))))
                }
                "dataApp" if items.len() == 3 => {
                    let f = parse_expr(&items[1])?;
                    let dt = parse_data_type(&items[2])?;
                    Ok(Expr::untyped(ExprNode::DataApp(Box::new(f), dt)))
                }
                "dataLam" if items.len() == 2 => {
                    let body = parse_expr(&items[1])?;
                    Ok(Expr::untyped(ExprNode::DataLambda(Box::new(body))))
                }
                "addrApp" if items.len() == 3 => {
                    let f = parse_expr(&items[1])?;
                    let addr = parse_address(&items[2])?;
                    Ok(Expr::untyped(ExprNode::AddrApp(Box::new(f), addr)))
                }
                "addrLam" if items.len() == 2 => {
                    let body = parse_expr(&items[1])?;
                    Ok(Expr::untyped(ExprNode::AddrLambda(Box::new(body))))
                }
                "natNatLam" if items.len() == 2 => {
                    let body = parse_expr(&items[1])?;
                    Ok(Expr::untyped(ExprNode::NatNatLambda(Box::new(body))))
                }
                "idxL" if items.len() == 3 => {
                    let i = parse_nat(&items[1])?;
                    let n = parse_nat(&items[2])?;
                    Ok(Expr::untyped(ExprNode::IndexLiteral(i, n)))
                }
                // Nat operations in expression position
                "natAdd" | "natMul" | "natPow" | "natMod" | "natFloorDiv" => {
                    let n = parse_nat(sexp)?;
                    Ok(Expr::untyped(ExprNode::NatLiteral(n)))
                }
                _ => Err(ParseError::Expr(format!("unknown expression form: {head}"))),
            }
        }
        Sexp::Empty => Err(ParseError::Expr("empty sexp".to_owned())),
    }
}

fn parse_expr_atom(s: &str) -> Result<ExprNode, ParseError> {
    // Variable: $e<index>
    if let Some(rest) = s.strip_prefix("$e") {
        let idx = rest
            .parse::<usize>()
            .map_err(|reason| ParseError::VarIndex {
                input: s.to_owned(),
                reason,
            })?;
        return Ok(ExprNode::Var(idx));
    }

    // Integer literal: <n>i
    if let Some(num) = s.strip_suffix('i')
        && let Ok(value) = num.parse::<i32>()
    {
        return Ok(ExprNode::Literal(LiteralData::Int(value)));
    }

    // Float literal: <n>f
    if let Some(num) = s.strip_suffix('f')
        && let Ok(value) = num.parse::<f32>()
    {
        return Ok(ExprNode::Literal(LiteralData::Float(OrderedFloat(value))));
    }

    // Double literal: <n>d
    if let Some(num) = s.strip_suffix('d')
        && let Ok(value) = num.parse::<f64>()
    {
        return Ok(ExprNode::Literal(LiteralData::Double(OrderedFloat(value))));
    }

    // Nat literal in expression position: <n>n
    if let Some(num) = s.strip_suffix('n')
        && let Ok(value) = num.parse::<i64>()
    {
        return Ok(ExprNode::NatLiteral(Nat::Cst(value)));
    }

    // Nat variable in expression position: $n<index>
    if let Some(rest) = s.strip_prefix("$n") {
        let idx = rest
            .parse::<usize>()
            .map_err(|reason| ParseError::VarIndex {
                input: s.to_owned(),
                reason,
            })?;
        return Ok(ExprNode::NatLiteral(Nat::Var(idx)));
    }

    // Boolean literals
    if s == "true" {
        return Ok(ExprNode::Literal(LiteralData::Bool(true)));
    }
    if s == "false" {
        return Ok(ExprNode::Literal(LiteralData::Bool(false)));
    }

    // // Float/Double without suffix (decimal number)
    // if (s.contains('.') || s.contains('E') || s.contains('e'))
    //     && let Ok(value) = s.parse::<f32>()
    // {
    //     return Ok(ExprNode::Literal(LiteralData::Float(OrderedFloat(value))));
    // }

    // Primitive
    Ok(ExprNode::Primitive(Primitive::from_name(s)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::rise::types::ScalarType;

    #[test]
    fn parse_simple_expr() {
        let expr: Expr = "$e0".parse().unwrap();
        assert_eq!(expr.node, ExprNode::Var(0));
        assert_eq!(expr.ty, None);
    }

    #[test]
    fn parse_typed_expr() {
        let expr: Expr = "(typeOf $e0 f32)".parse().unwrap();
        assert_eq!(expr.node, ExprNode::Var(0));
        assert_eq!(expr.ty, Some(Type::Data(DataType::Scalar(ScalarType::F32))));
    }

    #[test]
    fn parse_app() {
        let expr: Expr = "(app map (lam $e0))".parse().unwrap();
        match &expr.node {
            ExprNode::App(f, e) => {
                assert!(matches!(f.node, ExprNode::Primitive(Primitive::Map)));
                assert!(matches!(&e.node, ExprNode::Lambda(_)));
            }
            _ => panic!("expected app"),
        }
    }

    #[test]
    fn parse_nat_lambda() {
        let expr: Expr = "(natLam (natLam $e0))".parse().unwrap();
        match &expr.node {
            ExprNode::NatLambda(body) => match &body.node {
                ExprNode::NatLambda(inner) => {
                    assert!(matches!(inner.node, ExprNode::Var(0)));
                }
                _ => panic!("expected inner natLam"),
            },
            _ => panic!("expected natLam"),
        }
    }

    #[test]
    fn parse_complex_typed_expr() {
        let input = "(typeOf (lam (typeOf $e0 f32)) (fun f32 f32))";
        let expr: Expr = input.parse().unwrap();

        // Check outer type
        assert_eq!(
            expr.ty,
            Some(Type::Fun(
                Box::new(Type::Data(DataType::Scalar(ScalarType::F32))),
                Box::new(Type::Data(DataType::Scalar(ScalarType::F32))),
            ))
        );

        // Check it's a lambda
        match &expr.node {
            ExprNode::Lambda(body) => {
                assert_eq!(body.node, ExprNode::Var(0));
                assert_eq!(body.ty, Some(Type::Data(DataType::Scalar(ScalarType::F32))));
            }
            _ => panic!("expected lambda"),
        }
    }

    #[test]
    fn sexp_roundtrip_expr() {
        let expr = Expr::typed(
            ExprNode::Lambda(Box::new(Expr::typed(
                ExprNode::Var(0),
                Type::Data(DataType::Scalar(ScalarType::F32)),
            ))),
            Type::fun(
                Type::Data(DataType::Scalar(ScalarType::F32)),
                Type::Data(DataType::Scalar(ScalarType::F32)),
            ),
        );
        let sexp = expr.into_sexp().to_string();
        let parsed: Expr = sexp.parse().unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn to_tree_simple() {
        let expr: Expr = "(app map (lam $e0))".parse().unwrap();
        let tree = expr.to_typed_tree();

        assert_eq!(tree.label(), &RiseLabel::App);
        assert_eq!(tree.children().len(), 2);
        assert_eq!(
            tree.children()[0].label(),
            &RiseLabel::Primitive(Primitive::Map)
        );
        assert_eq!(tree.children()[1].label(), &RiseLabel::Lambda);
        assert_eq!(tree.children()[1].children()[0].label(), &RiseLabel::Var(0));
    }

    #[test]
    fn to_tree_typed() {
        let expr: Expr = "(typeOf $e0 f32)".parse().unwrap();
        let tree = expr.to_typed_tree();

        assert_eq!(tree.label(), &RiseLabel::TypeOf);
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), &RiseLabel::Var(0));
        assert_eq!(
            tree.children()[1].label(),
            &RiseLabel::Scalar(ScalarType::F32)
        );
    }

    #[test]
    fn to_tree_nested_types() {
        let expr: Expr = "(typeOf (lam (typeOf $e0 f32)) (fun f32 f32))"
            .parse()
            .unwrap();
        let tree = expr.to_typed_tree();

        assert_eq!(tree.label(), &RiseLabel::TypeOf);
        assert_eq!(tree.children()[0].label(), &RiseLabel::Lambda);

        // Inner lambda body should be typeOf
        let lam_body = &tree.children()[0].children()[0];
        assert_eq!(lam_body.label(), &RiseLabel::TypeOf);
        assert_eq!(lam_body.children()[0].label(), &RiseLabel::Var(0));
        assert_eq!(
            lam_body.children()[1].label(),
            &RiseLabel::Scalar(ScalarType::F32)
        );

        // Outer type should be (fun f32 f32)
        let outer_type = &tree.children()[1];
        assert_eq!(outer_type.label(), &RiseLabel::Fun);
        assert_eq!(
            outer_type.children()[0].label(),
            &RiseLabel::Scalar(ScalarType::F32)
        );
        assert_eq!(
            outer_type.children()[1].label(),
            &RiseLabel::Scalar(ScalarType::F32)
        );
    }

    #[test]
    fn parse_literal_int() {
        let expr: Expr = "42i".parse().unwrap();
        assert_eq!(expr.node, ExprNode::Literal(LiteralData::Int(42)));
    }

    #[test]
    fn parse_literal_float() {
        let expr: Expr = "3.11f".parse().unwrap();
        match expr.node {
            ExprNode::Literal(LiteralData::Float(f)) => {
                assert!((f.0 - 3.11).abs() < 0.01);
            }
            _ => panic!("expected float literal"),
        }
    }

    #[test]
    fn parse_literal_bool() {
        let expr_1: Expr = "true".parse().unwrap();
        assert_eq!(expr_1.node, ExprNode::Literal(LiteralData::Bool(true)));

        let expr_2: Expr = "false".parse().unwrap();
        assert_eq!(expr_2.node, ExprNode::Literal(LiteralData::Bool(false)));
    }

    #[test]
    fn parse_primitive() {
        let expr_1: Expr = "map".parse().unwrap();
        assert_eq!(expr_1.node, ExprNode::Primitive(Primitive::Map));

        let expr_2: Expr = "reduce".parse().unwrap();
        assert_eq!(expr_2.node, ExprNode::Primitive(Primitive::Reduce));
    }
}
