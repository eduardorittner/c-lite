use crate::lexer::Token;
use std::fmt::{Display, Write};

#[derive(Debug, Clone)]
pub enum ExternalDecl {
    Decl(Decl),
    FnDecl(FnDecl, StmtKind),
}

// <decl> ::= <var-decl> | <type-decl> | <struct-decl>
#[derive(Debug, Clone)]
pub struct Decl {
    pub token: Token,
    pub kind: DeclKind,
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    Var {
        ident: Token,
        spec: Option<TypeSpec>,
        init: Init,
    },
    Type {
        oldtype: TypeSpec,
        newtype: Token,
    },
    Struct {
        name: Token,
        fields: Vec<Field>,
    },
}

// TODO implement PrettyPrint for Field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Token,
    pub r#type: TypeSpec,
}

#[derive(Debug, Clone)]
pub struct TypeSpec {
    pub token: Token,
    pub kind: TypeSpecKind,
}

#[derive(Debug, Clone)]
pub enum TypeSpecKind {
    Type,
    UserType,
    Pointer,
}

#[derive(Debug, Clone)]
pub enum Init {
    Scalar(ExprKind),
    Aggregate(Vec<Init>),
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub token: Token,
    pub name: Token,
    pub params: Vec<Param>,
    pub ret: Option<TypeSpec>,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub spec: TypeSpec,
    pub name: Token,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Block {
        statements: Vec<StmtKind>,
    },
    Decl {
        declaration: Decl,
    },
    Expression {
        expression: ExprKind,
    },
    If {
        token: Token,
        cond: ExprKind,
        iftrue: Box<StmtKind>,
        else_token: Option<Token>,
        iffalse: Option<Box<StmtKind>>,
    },
    While {
        token: Token,
        cond: ExprKind,
        block: Box<StmtKind>,
    },
}

// Expressions are anything in the language that has an associated value
// or that performs some computation over other expressions
// Note that assignments such as "a = 1" are also (binary) expressions
#[derive(Debug, Clone)]
pub enum ExprKind {
    String(Token),
    Char(char),
    Number(u64),
    Bool(bool),
    Ident(Token),
    PostUnary {
        // a++ or a--
        op: Token,
        rhs: Box<ExprKind>,
    },
    Unary {
        // a++ or a--
        op: Token,
        rhs: Box<ExprKind>,
    },
    Binary {
        lhs: Box<ExprKind>,
        op: Token,
        rhs: Box<ExprKind>,
    },
    Postfix {
        lhs: Box<ExprKind>,
        op: Token,
    },
    Assign {
        lhs: Box<ExprKind>,
        op: Token,
        rhs: Box<ExprKind>,
    },
    CompoundAssign {
        lhs: Box<ExprKind>,
        op: Token,
        rhs: Box<ExprKind>,
    },
    Ternary {
        question: Box<ExprKind>,
        yes: Box<ExprKind>,
        no: Box<ExprKind>,
    },
    Comma {
        lhs: Box<ExprKind>,
        op: Token,
        rhs: Box<ExprKind>,
    },
}

/// Used for pretty-printing the ast in a more human-readable way
/// We can't just implement Display since we require extra state,
/// namely, how deep in the ast we are.
pub trait PrettyPrint {
    fn pretty_fmt(&self, depth: usize) -> String;

    fn indent_fmt(&self, depth: usize) -> String {
        let indent = "|".repeat(depth);
        format!("{}{}", indent, self.pretty_fmt(depth))
    }
}

impl PrettyPrint for ExternalDecl {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self {
            ExternalDecl::Decl(decl) => decl.indent_fmt(depth).to_string(),
            ExternalDecl::FnDecl(fn_decl, block) => {
                format!(
                    "{}{}",
                    fn_decl.indent_fmt(depth),
                    block.indent_fmt(depth + 1)
                )
            }
        }
    }
}

impl PrettyPrint for Decl {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self.kind {
            DeclKind::Var {
                ref ident,
                ref spec,
                ref init,
            } => {
                if let Some(spec) = spec {
                    format!(
                        "Var Declaration: '{}'\n{}\n{}",
                        ident.token,
                        spec.indent_fmt(depth + 1),
                        init.indent_fmt(depth + 1)
                    )
                } else {
                    format!(
                        "Var Declaration: '{}'\n{}",
                        ident.token,
                        init.indent_fmt(depth + 1)
                    )
                }
            }
            DeclKind::Type {
                oldtype: ref spec,
                ref newtype,
            } => {
                format!(
                    "Type Decl:\n{}\n{}newtype: {}",
                    spec.indent_fmt(depth + 1),
                    "|".repeat(depth + 1),
                    newtype.token
                )
            }
            DeclKind::Struct {
                ref name,
                ref fields,
            } => {
                format!(
                    "Struct Decl:\n{}{}, fields: {:?}",
                    "|".repeat(depth + 1),
                    name.token,
                    fields
                )
            }
        }
    }
}

impl PrettyPrint for Field {
    fn pretty_fmt(&self, _: usize) -> String {
        format!("Field {}: {}", self.name.token, self.r#type)
    }
}

impl PrettyPrint for TypeSpec {
    fn pretty_fmt(&self, _: usize) -> String {
        format!("type: {}", self.token.token)
    }
}

impl PrettyPrint for Init {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self {
            Init::Scalar(expr) => {
                format!("init:\n{}", expr.indent_fmt(depth + 1))
            }
            Init::Aggregate(inits) => {
                let mut result = String::from("init:\n");
                for init in inits {
                    let _ = write!(result, "{}, ", init.indent_fmt(depth + 1));
                }
                result
            }
        }
    }
}

impl PrettyPrint for FnDecl {
    fn pretty_fmt(&self, depth: usize) -> String {
        if let Some(ret) = &self.ret {
            format!(
                "Function {} -> {}:\n{}",
                self.name.token,
                ret,
                self.params.pretty_fmt(depth)
            )
        } else {
            format!(
                "Function {}:\n{}",
                self.name.token,
                self.params.pretty_fmt(depth)
            )
        }
    }
}

impl PrettyPrint for Param {
    fn pretty_fmt(&self, _: usize) -> String {
        format!("Param {}: {}", self.name.token, self.spec)
    }
}

impl PrettyPrint for Vec<Param> {
    fn pretty_fmt(&self, depth: usize) -> String {
        let mut result = String::new();
        for param in self {
            result += &(param.indent_fmt(depth + 1) + "\n");
        }
        result
    }
}

impl PrettyPrint for StmtKind {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self {
            StmtKind::Block { ref statements } => {
                let mut result = String::from("Statements:\n");
                for stmt in statements {
                    let _ = writeln!(result, "{}", stmt.indent_fmt(depth + 1));
                }
                result
            }
            StmtKind::Decl { ref declaration } => declaration.pretty_fmt(depth).to_string(),
            StmtKind::Expression { ref expression } => expression.pretty_fmt(depth).to_string(),
            StmtKind::If {
                token: _,
                else_token: _,
                ref cond,
                ref iftrue,
                ref iffalse,
            } => {
                if let Some(iffalse) = iffalse {
                    format!(
                        "If\n{}\n{}then:\n{}{}else:\n{}",
                        cond.indent_fmt(depth + 1),
                        "|".repeat(depth + 2),
                        iftrue.indent_fmt(depth + 3),
                        "|".repeat(depth + 2),
                        iffalse.indent_fmt(depth + 3)
                    )
                } else {
                    format!(
                        "If\n{}\n{}then:\n{}",
                        cond.indent_fmt(depth + 1),
                        "|".repeat(depth + 2),
                        iftrue.indent_fmt(depth + 3)
                    )
                }
            }
            StmtKind::While {
                token: _,
                ref cond,
                ref block,
            } => format!("While {cond}:\n{block}"),
        }
    }
}

impl PrettyPrint for ExprKind {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self {
            ExprKind::String(token) => {
                format!("String: {}", token.token.clone())
            }
            ExprKind::Char(c) => format!("'{c}'"),
            ExprKind::Number(n) => format!("'{n}'"),
            ExprKind::Bool(b) => format!("'{b}'"),
            ExprKind::Ident(token) => format!("'{}'", token.token),
            ExprKind::Unary { op, rhs } => {
                format!("Unary: {}\n{}", op.kind, rhs.indent_fmt(depth + 1))
            }
            ExprKind::Binary { lhs, op, rhs } => format!(
                "Binary: {}\n{}\n{}",
                op.kind,
                lhs.indent_fmt(depth + 1),
                rhs.indent_fmt(depth + 1)
            ),
            ExprKind::Postfix { lhs, op } => {
                format!("Postfix: {}\n{}", op.kind, lhs.indent_fmt(depth + 1))
            }
            ExprKind::Assign { lhs, rhs, .. } => format!(
                "Assign:\n{}\n{}",
                lhs.indent_fmt(depth + 1),
                rhs.indent_fmt(depth + 1)
            ),
            ExprKind::CompoundAssign { lhs, op, rhs } => format!(
                "CompoundAssign: {}\n{}\n{}",
                op.kind,
                lhs.indent_fmt(depth + 1),
                rhs.indent_fmt(depth + 1)
            ),
            ExprKind::Ternary { question, yes, no } => format!(
                "Ternary:\n{}\n{}\n{}",
                question.indent_fmt(depth + 1),
                yes.indent_fmt(depth + 1),
                no.indent_fmt(depth + 1)
            ),
            ExprKind::Comma { lhs, op, rhs } => format!(
                "Comma: {}\n{}\n{}",
                op.kind,
                lhs.indent_fmt(depth + 1),
                rhs.indent_fmt(depth + 1)
            ),
            ExprKind::PostUnary { op, rhs } => {
                format!("Post Unary: {}\n{}", op.kind, rhs.indent_fmt(depth + 1))
            }
        }
    }
}

macro_rules! display_impl {
    ($type:ty) => {
        impl Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.indent_fmt(0))
            }
        }
    };
}

display_impl!(ExternalDecl);
display_impl!(Decl);
display_impl!(Field);
display_impl!(TypeSpec);
display_impl!(Init);
display_impl!(FnDecl);
display_impl!(Param);
display_impl!(StmtKind);
display_impl!(ExprKind);
