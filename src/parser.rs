use crate::lexer::{Lexer, LiteralKind, Token, TokenKind};
use miette::Result;

mod ast {
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
    pub struct Field {
        pub name: Token,
        pub r#type: TypeSpec,
    }

    pub struct Stmt {
        token: Token,
        kind: StmtKind,
    }

    pub enum StmtKind {
        Compound {
            statements: Vec<Stmt>,
        },
        Decl {
            declaration: Decl,
        },
        Expression {
            expression: ExprKind,
        },
        If {
            cond: ExprKind,
            iftrue: Box<Stmt>,
            iffalse: Option<Box<Stmt>>,
        },
        While {
            cond: ExprKind,
            block: Box<Stmt>,
        },
    }

    pub trait PrettyPrint {
        fn pretty_fmt(&self, depth: usize) -> String;

        fn indent_fmt(&self, depth: usize) -> String {
            let indent = "|".repeat(depth);
            format!("{}{}", indent, self.pretty_fmt(depth))
        }
    }

    impl PrettyPrint for Decl {
        fn pretty_fmt(&self, _: usize) -> String {
            match self.kind {
                DeclKind::VarDecl {
                    ref ident,
                    ref spec,
                    ref init,
                } => {
                    format!(
                        "Declaration: var: {} type: {:?} value: {:?}",
                        ident.token, spec, init
                    )
                }
                DeclKind::TypeDecl {
                    oldtype: ref spec,
                    ref newtype,
                } => {
                    format!("Type Decl: type: {:?} newtype: {:?}", spec, newtype)
                }
                DeclKind::StructDecl {
                    ref name,
                    ref fields,
                } => {
                    format!("Struct Decl: name: {}, fields: {:?}", name.token, fields)
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
}

// Returns a Some(token) only if it matches the pattern
// Useful for not having to compare on enums without declaring their fields
// e.g. TokenKind::Literal(_)
macro_rules! match_next {
    ($parser:expr, $expected:pat) => {{
        let matched = match $parser.peek() {
            Some(token) => matches!(token.kind, $expected),
            None => false,
        };
        if matched {
            $parser.advance()
        } else {
            None
        }
    }};
}

use ast::*;

pub struct Parser<'src> {
    src: &'src str,
    lexer: Lexer<'src>,
    peeked: Option<Token>,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            lexer: Lexer::new(src),
            peeked: None,
        }
    }

    fn advance(&mut self) -> Option<Token> {
        if let Some(token) = self.peeked.clone() {
            self.peeked = None;
            Some(token)
        } else {
            self.lexer.next()
        }
    }

    fn peek(&mut self) -> Option<Token> {
        if let Some(token) = self.peeked.clone() {
            Some(token)
        } else {
            self.peeked = self.lexer.next();
            self.peeked.clone()
        }
    }

    fn expect(&mut self, expected: TokenKind, msg: &str) -> Result<Token> {
        self.expect_where(|t| t == expected, msg)
    }

    fn expect_where(
        &mut self,
        mut check: impl FnMut(TokenKind) -> bool,
        msg: &str,
    ) -> Result<Token> {
        match self.advance() {
            Some(token) if check(token.kind) => {
                Ok(token)
            },
            Some(token) => {
                Err(miette::miette! {
                    labels = vec![miette::LabeledSpan::at(token.offset..token.offset + token.token.len(), "here")],
                    help = format!("Unexpected token: {token:?}"),
                    "{msg}"
            }.with_source_code(self.src.to_string()))},
            None => Err(miette::miette! {
                        labels = vec![miette::LabeledSpan::at(self.lexer.offset - 1..self.lexer.offset, "here")],
                        "{msg}"
                    }.with_source_code(self.src.to_string())),
        }
    }

    pub fn parse(&mut self) -> Result<ExprKind> {
        self.expr()
    }

    pub fn stmt(&mut self) -> Result<Stmt> {
        todo!()
    }

    // <decl> ::= <var-decl> | <type-decl> | <struct-decl>
    // <var-decl> ::= let IDENT : <type-spec> = <init>
    //              | let IDENT = <init>
    // <type-decl> ::= type IDENT = IDENT
    // <struct-decl> ::= struct IDENT { <struct-field> }
    // <struct-field> ::= IDENT : TypeSpec ,
    pub fn decl(&mut self) -> Result<Decl> {
        if let Some(token) = self.peek().clone() {
            match token.kind {
                TokenKind::Type => self.type_decl(),
                TokenKind::Let => self.var_decl(),
                TokenKind::Struct => self.struct_decl(),
                _ => {
                    let kind = token.kind;
                    Err(miette::miette! {
                        labels = vec![miette::LabeledSpan::at(token.offset..token.offset + token.token.len(), "here")],
                        "Expected a declaration, got: {kind}"
                    }.with_source_code(self.src.to_string()))
                }
            }
        } else {
            unreachable!()
        }
    }

    pub fn var_decl(&mut self) -> Result<Decl> {
        let token = self.expect(TokenKind::Let, "Should have 'let' keyword here")?;
        let ident = self.expect(TokenKind::Ident, "Expected var name after 'let' keyword")?;
        let spec = if let Some(_colon) = match_next!(self, TokenKind::Colon) {
            Some(self.type_spec()?)
        } else {
            None
        };
        self.expect(TokenKind::Eq, "Expected '=' after var name")?;
        let init = self.initializer()?;
        Ok(Decl {
            token,
            kind: DeclKind::VarDecl { ident, spec, init },
        })
    }

    pub fn type_decl(&mut self) -> Result<Decl> {
        let token = self.expect(TokenKind::Type, "Should have type keyword")?;
        let newtype = self.expect(TokenKind::Ident, "Expected ident after 'type' keyword")?;
        self.expect(TokenKind::Eq, "Expected '='")?;
        if let Ok(oldtype) = self.type_spec() {
            Ok(Decl {
                token,
                kind: DeclKind::TypeDecl { oldtype, newtype },
            })
        } else {
            let wrong_token = self.advance().unwrap();
            let wrong_kind = wrong_token.kind;
            Err(miette::miette! {
                    labels = vec![miette::LabeledSpan::at(wrong_token.offset..wrong_token.offset+wrong_token.token.len(), "here")],
                    "Expected a valid type, got: {wrong_kind}",
                }.with_source_code(self.src.to_string()))
        }
    }

    // <struct-decl> ::= struct IDENT { <struct-field-list> }
    // <struct-field-list> ::= <struct-field-list> , <struct-field>
    // <struct-field> ::= IDENT : TypeSpec ,
    pub fn struct_decl(&mut self) -> Result<Decl> {
        let struct_token = self.expect(TokenKind::Struct, "unreachable")?;
        let name = self.expect(TokenKind::Ident, "Expected struct name")?;
        self.expect(TokenKind::OpenBrace, "Expected open brace")?;
        let mut fields = Vec::new();

        while let Some(field_name) = match_next!(self, TokenKind::Ident) {
            self.expect(TokenKind::Colon, "Expected colon ':' after field name")?;
            let spec = self.type_spec()?;
            fields.push(Field {
                name: field_name,
                r#type: spec,
            });

            if let Some(_) = match_next!(self, TokenKind::Comma) {
            } else {
                break;
            }
        }

        if fields.is_empty() {
            let token = self.advance().unwrap();
            return            Err(miette::miette! {
                    labels = vec![miette::LabeledSpan::at(token.offset..token.offset+token.token.len(), "here")],
                    "Expected a struct field identifier, got: {token:?}",
                }.with_source_code(self.src.to_string()));
        }

        self.expect(TokenKind::CloseBrace, "Expected close brace")?;
        Ok(Decl {
            token: struct_token,
            kind: DeclKind::StructDecl { name, fields },
        })
    }

    pub fn type_spec(&mut self) -> Result<TypeSpec> {
        if let Some(token) = match_next!(
            self,
            TokenKind::Int
                | TokenKind::UInt
                | TokenKind::Char
                | TokenKind::Void
                | TokenKind::Bool
                | TokenKind::Float
        ) {
            Ok(TypeSpec {
                token,
                kind: TypeSpecKind::Type,
            })
        } else if let Some(token) = match_next!(self, TokenKind::Ident) {
            Ok(TypeSpec {
                token,
                kind: TypeSpecKind::UserType,
            })
        } else {
            Err(miette::miette!(
                "Expected type spec, got: {:?}",
                self.peek().unwrap()
            ))
        }
    }

    pub fn initializer(&mut self) -> Result<Init> {
        // TODO parse struct initializers and so on
        let expr = self.conditional_expr()?;
        Ok(Init::Scalar(expr))
    }

    // <expr> ::= <assign-expr> | <expr> , <assign-expr>
    pub fn expr(&mut self) -> Result<ExprKind> {
        let expr = self.assign_expr()?;

        if let Some(token) = match_next!(self, TokenKind::Comma) {
            let rhs = self.assign_expr()?;
            Ok(ExprKind::Comma {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <assign-expr> ::= <conditional-expr>    | <unary-expr> assign-operator <assign-expr>
    // <assign-operator> ::= '=' | '*=' | '/=' | '%=' | '+=' | '-=' | '<<=' | '>>=' | '&=' | '^=' | '|='
    //
    // NOTE: a <conditional-expr> can be interpreted as <unary-expr>, so calling
    // self.conditional_expr() is sufficient
    pub fn assign_expr(&mut self) -> Result<ExprKind> {
        let expr = self.conditional_expr()?;

        if let Some(token) = match_next!(self, TokenKind::Eq) {
            let rhs = self.assign_expr()?;
            Ok(ExprKind::Assign {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else if let Some(token) = match_next!(
            self,
            TokenKind::StarEq | TokenKind::SlashEq | TokenKind::PlusEq | TokenKind::MinusEq
        ) {
            let rhs = self.assign_expr()?;
            Ok(ExprKind::CompoundAssign {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <conditional-expr> ::= <logical-or-expr> | <logical-or-expr> '?' <expr> ':' <conditional-expr>
    pub fn conditional_expr(&mut self) -> Result<ExprKind> {
        let expr = self.or()?;

        if let Some(_) = match_next!(self, TokenKind::Question) {
            let yes = self.expr()?;
            self.expect(
                TokenKind::Colon,
                "Expected colon after second ternary clause",
            )?;
            let no = self.conditional_expr()?;
            Ok(ExprKind::Ternary {
                question: Box::new(expr),
                yes: Box::new(yes),
                no: Box::new(no),
            })
        } else {
            Ok(expr)
        }
    }

    // <logical-or-expr> ::= <logical-and-expr> | <logical-or-expr> '||' <logical-and-expr>
    pub fn or(&mut self) -> Result<ExprKind> {
        let expr = self.and()?;

        if let Some(token) = match_next!(self, TokenKind::OrOr) {
            let rhs = self.or()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <logical-and-expr> ::= <bit-or-expr>     | <logical-and-expr> '&&' <bit-and-expr>
    pub fn and(&mut self) -> Result<ExprKind> {
        let expr = self.bit_or()?;

        if let Some(token) = match_next!(self, TokenKind::AndAnd) {
            let rhs = self.and()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <bit-or-expr> ::= <bit-xor-expr>         | <bit-or-expr> '|' <bit-xor-expr>
    pub fn bit_or(&mut self) -> Result<ExprKind> {
        let expr = self.bit_xor()?;

        if let Some(token) = match_next!(self, TokenKind::Or) {
            let rhs = self.bit_or()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <bit-xor-expr> ::= <bit-and-expr>        | <bit-xor-expr> '^' <bit-and-expr>
    pub fn bit_xor(&mut self) -> Result<ExprKind> {
        let expr = self.bit_and()?;

        if let Some(token) = match_next!(self, TokenKind::Xor) {
            let rhs = self.bit_xor()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <bit-and-expr> ::= <equality-expr>       | <bit-and-expr> '&' <equality-expr>
    pub fn bit_and(&mut self) -> Result<ExprKind> {
        let expr = self.equality()?;

        if let Some(token) = match_next!(self, TokenKind::And) {
            let rhs = self.bit_and()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <equality-expr> ::= <relational-expr>    | <equality-expr> '==' <relational-expr>
    //                                          | <equality-expr> '!=' <relational-expr>
    pub fn equality(&mut self) -> Result<ExprKind> {
        let expr = self.comparison()?;

        if let Some(token) = match_next!(self, TokenKind::EqEq | TokenKind::NotEq) {
            let rhs = self.equality()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <comparison-expr> ::= <shift-expr>       | <comparison-expr> '>' <shift-expr>
    //                                          | <comparison-expr> '<' <shift-expr>
    //                                          | <comparison-expr> '>=' <shift-expr>
    //                                          | <comparison-expr> '<=' <shift-expr>
    pub fn comparison(&mut self) -> Result<ExprKind> {
        let expr = self.shift()?;

        if let Some(token) = match_next!(
            self,
            TokenKind::Greater | TokenKind::Smaller | TokenKind::GreaterEq | TokenKind::SmallerEq
        ) {
            let rhs = self.comparison()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <shift-expr> ::= <add-expr>              | <shift-expr> '>>' <addsub-expr>
    //                                          | <shift-expr> '<<' <addsub-expr>
    pub fn shift(&mut self) -> Result<ExprKind> {
        let expr = self.addsub()?;

        if let Some(token) =
            match_next!(self, TokenKind::GreaterGreater | TokenKind::SmallerSmaller)
        {
            let rhs = self.shift()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <addsub-expr> ::= <muldiv-expr>          | <addsub-expr> '+' <muldiv-expr>
    //                                          | <addsub-expr> '-' <muldiv-expr>
    pub fn addsub(&mut self) -> Result<ExprKind> {
        let expr = self.muldiv()?;

        if let Some(token) = match_next!(self, TokenKind::Plus | TokenKind::Minus) {
            let rhs = self.muldiv()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <muldiv-expr> ::= <unary-expr>            | <muldiv-expr> '*' <unary-expr>
    //                                          | <muldiv-expr> '/' <unary-expr>
    //                                          | <muldiv-expr> '%' <unary-expr>
    pub fn muldiv(&mut self) -> Result<ExprKind> {
        let expr = self.unary_expr()?;

        if let Some(token) = match_next!(
            self,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent
        ) {
            let rhs = self.muldiv()?;
            Ok(ExprKind::Binary {
                lhs: Box::new(expr),
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            Ok(expr)
        }
    }

    // <unary-expr> ::= <postfix-expr>
    //                | ++ <unary-expr>
    //                | -- <unary-expr>
    //                | <unary-operator> <unary-expr>
    //                | sizeof <unary-expr>
    //                | sizeof ( <type-name> )
    //
    // <unary-operator> ::= & | * | + | - | ~ | !
    pub fn unary_expr(&mut self) -> Result<ExprKind> {
        if let Some(token) = match_next!(self, TokenKind::PlusPlus | TokenKind::MinusMinus) {
            // ++a | --a desugars to a += 1 | a -= 1
            let rhs = self.unary_expr()?;

            Ok(ExprKind::CompoundAssign {
                lhs: Box::new(rhs),
                op: token,
                rhs: Box::new(ExprKind::Number(1)),
            })
        } else if let Some(token) = match_next!(
            self,
            TokenKind::And
                | TokenKind::Not
                | TokenKind::Star
                | TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Tilde
        ) {
            let rhs = self.unary_expr()?;

            Ok(ExprKind::Unary {
                op: token,
                rhs: Box::new(rhs),
            })
        } else {
            self.postfix_expr()
        }
    }

    // <postfix-expr> ::= <primary-expression>
    //                  | <postfix-expr> [ <expr> ]
    //                  | <postfix-expr> ( <arg-list> )
    //                  | <postfix-expr> . identifier
    //                  | <postfix-expr> -> identifier
    //                  | <postfix-expr> ++
    //                  | <postfix-expr> --
    //                  | ( <type-name> ) { <initializer-list> }
    //                  | ( <type-name> ) { <initializer-list> , }
    pub fn postfix_expr(&mut self) -> Result<ExprKind> {
        let mut expr = self.primary_expr()?;
        // TODO parse ( type-name ) {initializer-list} syntax

        while let Some(token) = match_next!(
            self,
            TokenKind::OpenBrace
                | TokenKind::OpenParen
                | TokenKind::PlusPlus
                | TokenKind::MinusMinus
                | TokenKind::Dot
                | TokenKind::Arrow
        ) {
            match token.kind {
                TokenKind::OpenBrace => {
                    let index = self.expr()?;
                    self.expect(
                        TokenKind::CloseBrace,
                        "Expected closing brace ']' after index expression",
                    )?;
                    // TODO expr = index_desugar(expr, index)
                    let expr = todo!();
                }
                TokenKind::OpenParen => {
                    // TODO parse args-list
                    let expr = todo!();
                }
                TokenKind::PlusPlus | TokenKind::MinusMinus => {
                    expr = ExprKind::Postfix {
                        lhs: Box::new(expr),
                        op: token,
                    };
                }
                // TODO handle '->' and '.'
                _ => unreachable!(),
            }
        }

        Ok(expr)
    }

    // <primary-expr> ::= identifier | <constant> | ( <expr> )
    pub fn primary_expr(&mut self) -> Result<ExprKind> {
        let token = self
            .peek()
            .expect("Should have token when parsing constant");

        match token.kind {
            TokenKind::Ident => {
                self.advance();
                Ok(ExprKind::Ident(token))
            }
            TokenKind::Literal(_) | TokenKind::True | TokenKind::False => self.constant(),
            TokenKind::OpenParen => {
                self.advance();
                let expr = self.expr()?;
                self.expect(
                    TokenKind::CloseParen,
                    "Expected closing paren ')' after expression",
                )?;
                Ok(expr)
            }
            _ => unreachable!(),
        }
    }

    // <constant> ::= NUMBER | CHAR | STRING | true | false
    pub fn constant(&mut self) -> Result<ExprKind> {
        let token = self
            .peek()
            .expect("Should have token when parsing constant");

        match token.kind {
            TokenKind::Literal(LiteralKind::Decimal) => self.decimal_number(),
            TokenKind::Literal(LiteralKind::Hex) => self.hex_number(),
            TokenKind::Literal(LiteralKind::Binary) => self.binary_number(),
            TokenKind::Literal(LiteralKind::String) => self.string(),
            TokenKind::True => Ok(ExprKind::Bool(true)),
            TokenKind::False => Ok(ExprKind::Bool(false)),
            _ => unreachable!(),
        }
    }

    pub fn string(&mut self) -> Result<ExprKind> {
        todo!()
    }

    fn decimal_number(&mut self) -> Result<ExprKind> {
        let token = self
            .advance()
            .expect("Should have token when parsing decimal literal");
        assert!(token.kind == TokenKind::Literal(LiteralKind::Decimal));

        let digits = token.token.replace("_", "").replace("`", "");
        if let Ok(n) = digits.parse::<u64>() {
            Ok(ExprKind::Number(n))
        } else {
            Err(miette::miette! {
                labels = vec![miette::LabeledSpan::at(token.offset..token.offset + token.token.len(), "here")],
                "Invalid numeric literal"
            }.with_source_code(self.src.to_string()))
        }
    }

    fn binary_number(&mut self) -> Result<ExprKind> {
        let token = self
            .advance()
            .expect("Should have token when parsing binary literal");
        assert!(token.kind == TokenKind::Literal(LiteralKind::Binary));

        let digits = token.token.strip_prefix("0b").unwrap();

        if let Ok(n) = u64::from_str_radix(&digits, 2) {
            Ok(ExprKind::Number(n))
        } else {
            Err(miette::miette! {
            labels =  vec![miette::LabeledSpan::at(token.offset..token.offset + token.token.len(), "here")],
            "Invalid binary literal"
            }.with_source_code(self.src.to_string()))
        }
    }

    fn hex_number(&mut self) -> Result<ExprKind> {
        let token = self
            .advance()
            .expect("Should have token when parsing hex literal");
        assert!(token.kind == TokenKind::Literal(LiteralKind::Hex));

        let digits = token.token.strip_prefix("0x").unwrap();

        if let Ok(n) = u64::from_str_radix(&digits, 16) {
            Ok(ExprKind::Number(n))
        } else {
            Err(miette::miette! {
            labels =  vec![miette::LabeledSpan::at(token.offset..token.offset + token.token.len(), "here")],
            "Invalid hexadecimal literal"
            }.with_source_code(self.src.to_string()))
        }
    }
}

#[cfg(test)]
pub mod tests {

    use crate::Parser;
    use insta::assert_debug_snapshot;

    fn parse_expr(expr: &str) -> String {
        let mut parser = Parser::new(&expr);
        parser.expr().unwrap().to_string()
    }

    #[test]
    fn unary_prec() {
        let source = "+2 + 3";
        let expected = "Binary: '+'\n|Unary: '+'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn mul_prec() {
        let source = "1 + 2 * 3";
        let expected = "Binary: '+'\n|'1'\n|Binary: '*'\n||'2'\n||'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn add_prec() {
        let source = "1 + 2 >> 3";
        let expected = "Binary: '>>'\n|Binary: '+'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn shift_prec() {
        let source = "1 >> 2 < 3";
        let expected = "Binary: '<'\n|Binary: '>>'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn comparison_prec() {
        let source = "1 > 2 == 3";
        let expected = "Binary: '=='\n|Binary: '>'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn equal_prec() {
        let source = "1 == 2 & 3";
        let expected = "Binary: '&'\n|Binary: '=='\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn bit_and_prec() {
        let source = "1 & 2 ^ 3";
        let expected = "Binary: '^'\n|Binary: '&'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn bit_xor_prec() {
        let source = "1 ^ 2 | 3";
        let expected = "Binary: '|'\n|Binary: '^'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn bit_or_prec() {
        let source = "1 | 2 && 3";
        let expected = "Binary: '&&'\n|Binary: '|'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn and_prec() {
        let source = "1 && 2 || 3";
        let expected = "Binary: '||'\n|Binary: '&&'\n||'1'\n||'2'\n|'3'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn or_prec() {
        let source = "1 || 2 ? 3 : 4";
        let expected = "Ternary:\n|Binary: '||'\n||'1'\n||'2'\n|'3'\n|'4'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn unary_and_deref() {
        let source = "a = *p++";
        let expected = "Assign:\n|'a'\n|Unary: '*'\n||Postfix: '++'\n|||'p'";
        assert_eq!(expected, parse_expr(source));
    }

    #[test]
    fn var_decl_with_type() {
        let source = "let a: int = 202";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn var_decl_no_type() {
        let source = "let a = 202";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn var_decl_complicated_expr() {
        let source = "let a = 1 + 2 * 3 << 4 + (2)";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn var_decl_unary_expr() {
        let source = "let a = !1";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn type_decl() {
        let source = "type mytype = int";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn type_decl_usertype() {
        let source = "type mytype = myusertype";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn type_decl_expr() {
        // Assert that we get an error
        let source = "type wrongtype = 1";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn invalid_decl_start() {
        let source = "void wrongtype = 1";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn empty_decl() {
        let source = "";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn struct_decl() {
        let source = "struct a { field: void }";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn composite_struct_decl() {
        let source = "struct b { field: void, this: char}";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn wrong_struct_decl() {
        let source = "struct b {}";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }

    #[test]
    fn struct_name_as_type() {
        let source = "struct void {a: char}";
        let mut parser = Parser::new(&source);
        let result = parser.decl();
        assert_debug_snapshot!(result)
    }
}
