use crate::lexer::{Lexer, LiteralKind, Token, TokenKind};
use miette::Result;

// Taken from [Gnu C Reference](https://www.gnu.org/software/gnu-c-manual/gnu-c-manual.html#Operator-Precedence)
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Precedence {
    // derive(Ord) makes the first variant the highest and so on
    Primary,
    Call, // (), [], .field
    Unary,
    MulDiv, // '*' | '/' | '%'
    AddSub, // '+' | '-'
    BitShift,
    Inequality, // '>' | '<' | '>=' | '<='
    Equality,   // '=' | '!='
    BitAnd,
    BitXor,
    BitOr,
    LogicAnd,
    LogicOr,
    Ternary,
    Assign,
    Comma,
    None,
}

mod ast {
    use crate::lexer::Token;

    enum Node {}

    #[derive(Debug, Clone)]
    pub enum ExprKind {
        String(Token),
        Char(char),
        Number(u64),
        Bool(bool),
        Ident(Token),
        Unary {
            op: Token,
            rhs: Box<ExprKind>,
        },
        Binary {
            lhs: Box<ExprKind>,
            op: Token,
            rhs: Box<ExprKind>,
        },
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
    ast: Vec<ExprKind>,
    lexer: Lexer<'src>,
    peeked: Option<Token>,
}

impl<'src> Parser<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            ast: Vec::new(),
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

    fn consume(&mut self, expected: TokenKind, msg: &str) -> Result<()> {
        self.expect(expected, msg).map(|_| ())
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

    pub fn parse(&'src mut self) -> Result<ExprKind> {
        self.expr(Precedence::Assign)
    }

    pub fn expr(&mut self) -> Result<ExprKind> {
        todo!()
    }

    // Consumes an expression until it finds a operator with lower precedence
    // than the previous one.
    // e.g. when parsing "a + b * c" we parse a + (b * c) but when parsing
    // "a * b + c" we return (a * b) and let expr() parse the remaining "+"
    // expression
    pub fn inner_expr(&mut self, prec: Precedence) -> Result<ExprKind> {
        let token = self.peek().expect("Expected expression");

        // Parse infix expression
        let mut lhs = match token.kind {
            TokenKind::OpenParen => self.grouping(),
            _ => unreachable!(),
        }?;

        loop {
            let current_prec = self.peek().unwrap().infix_prec()?;
            if prec <= current_prec {
                break;
            }

            let token = self.advance().unwrap();
            lhs = match token.kind {
                TokenKind::Plus | TokenKind::Minus => self.binary(lhs)?,
                _ => unreachable!(),
            }
        }

        Ok(ExprKind::Binary {
            lhs: Box::new(lhs),
            op: token,
            rhs: Box::new(rhs),
        })
    }

    // prefix_expr
    pub fn prefix_expr(&mut self) -> Result<ExprKind> {
        let token = self.advance().unwrap();
        let expr = match token.kind {
            TokenKind::OpenParen => self.grouping(),
            _ => unreachable!(),
        }?;
        Ok(expr)
    }

    pub fn grouping(&mut self) -> Result<ExprKind> {
        let expr = self.expr(Precedence::Assign)?;
        self.expect(
            TokenKind::CloseParen,
            "Expected closing paren ')' after expression",
        )?;
        Ok(expr)
    }

    pub fn binary(&mut self, lhs: ExprKind) -> Result<ExprKind> {
        let token = self.advance().unwrap();
        let prec = match token.kind {
            TokenKind::Plus | TokenKind::Minus => Precedence::AddSub,
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => Precedence::MulDiv,
            // TODO add other cases
            _ => unreachable!(),
        };

        let expr = self.expr(prec)?;

        Ok(ExprKind::Binary {
            lhs: Box::new(lhs),
            op: token,
            rhs: Box::new(expr),
        })
    }

    pub fn unary(&mut self) -> Result<ExprKind> {
        let token = self.advance().unwrap();

        let expr = self.expr(Precedence::Unary)?;

        Ok(ExprKind::Unary {
            op: token,
            rhs: Box::new(expr),
        })
    }

    // <immutable> ::= ( <exp> ) | <call> | <constant>
    pub fn immutable(&mut self) -> Result<ExprKind> {
        let token = self.peek().unwrap();

        let expr = match token.kind {
            TokenKind::OpenParen => {
                self.advance();
                let expr = self.expr()?;
                
            },
            _ => unreachable!()
        }

        
        todo!()
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
