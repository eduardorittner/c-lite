
use std::fmt::Write;

use crate::lexer::Token;

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

// <decl> ::= <var-decl> | <type-decl> | <struct-decl>
#[derive(Debug, Clone)]
pub struct Decl {
    pub token: Token,
    pub kind: DeclKind,
}

#[derive(Debug, Clone)]
pub enum DeclKind {
    VarDecl {
        ident: Token,
        spec: Option<TypeSpec>,
        init: Init,
    },
    TypeDecl {
        oldtype: TypeSpec,
        newtype: Token,
    },
    StructDecl {
        name: Token,
        fields: Vec<Field>,
    },
}

#[derive(Debug, Clone)]
pub enum Init {
    Scalar(ExprKind),
    Aggregate(Vec<Box<Init>>),
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

impl std::fmt::Display for Init {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.indent_fmt(0))
    }
}

#[derive(Debug, Clone)]
pub struct TypeSpec {
    pub token: Token,
    pub kind: TypeSpecKind,
}

impl PrettyPrint for TypeSpec {
    fn pretty_fmt(&self, depth: usize) -> String {
        format!("type: {}", self.token.token)
    }
}

impl std::fmt::Display for TypeSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.indent_fmt(0))
    }
}

#[derive(Debug, Clone)]
pub enum TypeSpecKind {
    Type,
    UserType,
    Pointer,
}

// TODO implement PrettyPrint for Field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: Token,
    pub r#type: TypeSpec,
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

#[derive(Debug, Clone)]
pub enum ExternalDecl {
    Decl(Decl),
    FnDecl(FnDecl, Vec<StmtKind>),
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

pub trait PrettyPrint {
    fn pretty_fmt(&self, depth: usize) -> String;

    fn indent_fmt(&self, depth: usize) -> String {
        let indent = "|".repeat(depth);
        format!("{}{}", indent, self.pretty_fmt(depth))
    }
}

impl PrettyPrint for StmtKind {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self {
            StmtKind::Block { ref statements } => {
                let mut result = String::from("Statements:\n");
                for stmt in statements {
                    let _ = write!(result, "{}\n", stmt.indent_fmt(depth + 1));
                }
                result
            }
            StmtKind::Decl { ref declaration } => {
                format!("{}", declaration.pretty_fmt(depth))
            }
            StmtKind::Expression { ref expression } => {
                format!("{}", expression.pretty_fmt(depth))
            }
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

impl std::fmt::Display for StmtKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.indent_fmt(0))
    }
}

impl PrettyPrint for Decl {
    fn pretty_fmt(&self, depth: usize) -> String {
        match self.kind {
            DeclKind::VarDecl {
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
            DeclKind::TypeDecl {
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
            DeclKind::StructDecl {
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

impl std::fmt::Display for Decl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.indent_fmt(0))
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
                todo!("Post Unary: {}\n{}", op.kind, rhs.indent_fmt(depth + 1))
            }
        }
    }
}

impl std::fmt::Display for ExprKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.indent_fmt(0))
    }
}
