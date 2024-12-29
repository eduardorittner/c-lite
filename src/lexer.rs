#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub token: String,
    pub offset: usize,
}

use crate::parser::Precedence;
use miette::Result;
impl Token {
    pub fn infix_prec(&self) -> Result<Precedence> {
        match self.kind {
            TokenKind::Plus | TokenKind::Minus => Ok(Precedence::AddSub),
            _ => Err(miette::miette! {
            labels = vec![miette::LabeledSpan::at(self.offset..self.offset + self.token.len(), "here")],
            help = format!("Unexpected token: {self:?}"),
            "Not a valid infix operator token"}),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Ident,
    Literal(LiteralKind),
    Whitespace,

    // Operators
    Plus,
    PlusEq,
    Minus,
    MinusEq,
    Slash,
    SlashEq,
    Star,
    StarEq,
    Percent,
    Eq,
    Not,
    And,
    Or,

    // Comparison Operators
    EqEq,
    NotEq,
    AndAnd,
    OrOr,
    GreaterEq,
    Greater,
    SmallerEq,
    Smaller,

    // Separators
    OpenParen,
    CloseParen,
    OpenBracket,  // `[`
    CloseBracket, // `]`
    OpenBrace,    // `{`
    CloseBrace,   // `}`
    Comma,        // `,`
    Dot,          // `.`
    Colon,        // `:`
    Semicolon,    // `;`

    // boolean
    True,
    False,

    // Control-flow
    For,
    If,
    Return,
    While,

    // Data types
    Enum,
    Struct,
    Union,
    TagUnion,
    Type,

    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralKind {
    Hex,
    Decimal,
    Binary,
    String,
}

pub struct Lexer<'src> {
    src: &'src str,
    rest: &'src str,
    pub offset: usize,
    finished: bool,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            rest: src,
            offset: 0,
            finished: false,
        }
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let mut chars = self.rest.chars();
            let char = match chars.next() {
                Some(c) => c,
                None => {
                    self.finished = true;
                    return Some(Token {
                        kind: Eof,
                        token: String::new(),
                        offset: self.offset,
                    });
                }
            };
            let char_str = &self.rest[..char.len_utf8()];
            let char_onwards = self.rest;
            let char_offset = self.offset;

            self.rest = chars.as_str();
            self.offset += char.len_utf8();

            let simple_token = move |kind: TokenKind| {
                Some(Token {
                    kind,
                    token: char_str.to_string(),
                    offset: char_offset,
                })
            };

            enum State {
                Ident,
                Whitespace,
                NumberLit,
                StringLit,
                IfEqElse(char, TokenKind, TokenKind),
            }
            use TokenKind::*;

            let state = match char {
                '.' => return simple_token(Dot),
                ',' => return simple_token(Comma),
                ';' => return simple_token(Semicolon),
                ':' => return simple_token(Colon),
                '(' => return simple_token(OpenParen),
                ')' => return simple_token(CloseParen),
                '[' => return simple_token(OpenBracket),
                ']' => return simple_token(CloseBracket),
                '{' => return simple_token(OpenBrace),
                '}' => return simple_token(CloseBrace),
                '%' => return simple_token(Percent),
                '+' => State::IfEqElse('=', PlusEq, Plus),
                '-' => State::IfEqElse('=', MinusEq, Minus),
                '*' => State::IfEqElse('=', StarEq, Star),
                '/' => State::IfEqElse('=', SlashEq, Slash),
                '=' => State::IfEqElse('=', EqEq, Eq),
                '!' => State::IfEqElse('=', NotEq, Not),
                '>' => State::IfEqElse('=', GreaterEq, Greater),
                '<' => State::IfEqElse('=', SmallerEq, Smaller),
                '&' => State::IfEqElse('&', AndAnd, And),
                '|' => State::IfEqElse('|', OrOr, Or),
                '"' => State::StringLit,
                '\'' => State::StringLit,
                '0'..='9' => State::NumberLit,
                'a'..='z' | 'A'..='Z' | '_' => State::Ident,
                c if c.is_whitespace() => State::Whitespace,
                c => panic!("Unexpected character during lexing: '{c}'"),
            };

            match state {
                State::Whitespace => {
                    let first_non_whitespace = char_onwards
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or(char_onwards.len());

                    let whitespace = &char_onwards[..first_non_whitespace];
                    let whitespace_offset = whitespace.len() - char.len_utf8();
                    let token = Some(Token {
                        kind: TokenKind::Whitespace,
                        token: char_onwards[..whitespace.len()].to_string(),
                        offset: char_offset,
                    });

                    self.offset += whitespace_offset;
                    self.rest = &self.rest[whitespace_offset..];

                    return token;
                }
                State::Ident => {
                    let first_non_ident = char_onwards
                        .find(|c| !matches!(c,'a'..='z'| 'A'..='Z'| '_'| '0'..='9'))
                        .unwrap_or(char_onwards.len());

                    let lit = &char_onwards[..first_non_ident];
                    let lit_offset = lit.len() - char.len_utf8();

                    let kind = match lit {
                        "for" => For,
                        "if" => If,
                        "return" => Return,
                        "while" => While,
                        "enum" => Enum,
                        "struct" => Struct,
                        "union" => Union,
                        "tagunion" => TagUnion,
                        "type" => Type,
                        "true" => True,
                        "false" => False,
                        _ => Ident,
                    };

                    let token = Some(Token {
                        kind,
                        token: char_onwards[..lit.len()].to_string(),
                        offset: char_offset,
                    });

                    self.offset += lit_offset;
                    self.rest = &self.rest[lit_offset..];
                    return token;
                }
                State::IfEqElse(char, yes, no) => {
                    if self.rest.starts_with(char) {
                        let token = char_onwards[..char_str.len() + char.len_utf8()].to_string();
                        self.offset += char.len_utf8();
                        self.rest = &self.rest[char.len_utf8()..];
                        return Some(Token {
                            kind: yes,
                            token,
                            offset: char_offset,
                        });
                    } else {
                        return Some(Token {
                            kind: no,
                            token: char_str.to_string(),
                            offset: char_offset,
                        });
                    }
                }
                State::NumberLit => {
                    let (prefix, lit_kind) = match char_onwards {
                        c if c.starts_with("0x") => (2, LiteralKind::Hex),
                        c if c.starts_with("0b") => (2, LiteralKind::Binary),
                        _ => (0, LiteralKind::Decimal),
                    };

                    let first_non_number = match lit_kind {
                        LiteralKind::Decimal => {
                            char_onwards[prefix..].find(|c| !matches!(c, '0'..='9' | '_' | '`'))
                        }
                        LiteralKind::Binary => {
                            char_onwards[prefix..].find(|c| !matches!(c, '0' | '1'))
                        }
                        LiteralKind::Hex => char_onwards[prefix..]
                            .find(|c| !matches!(c, '0'..='9' | 'a'..='f' | 'A'..='F' | '_' | '`')),
                        _ => unreachable!(),
                    };

                    let first_non_number = match first_non_number {
                        Some(n) => n + prefix,
                        None => char_onwards.len(),
                    };

                    let lit = &char_onwards[..first_non_number];
                    let lit_offset = lit.len() - char.len_utf8();
                    let token = Some(Token {
                        kind: TokenKind::Literal(lit_kind),
                        token: char_onwards[..lit.len()].to_string(),
                        offset: char_offset,
                    });
                    self.offset += lit_offset;
                    self.rest = &self.rest[lit_offset..];

                    return token;
                }
                State::StringLit => {
                    let closing_quote = char_onwards
                        .char_indices()
                        .find(|(i, c)| {
                            *i != 0
                                && *c == char
                                && (*i == 0 || char_onwards.chars().nth(i - 1).unwrap() != '\\')
                        })
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(char_onwards.len());

                    let lit = &char_onwards[..closing_quote];

                    let lit_offset = lit.len() - char.len_utf8();
                    let token = Some(Token {
                        kind: TokenKind::Literal(LiteralKind::String),
                        token: char_onwards[..lit.len()].to_string(),
                        offset: char_offset,
                    });
                    self.offset += lit_offset;
                    self.rest = &self.rest[lit_offset..];
                    return token;
                }
            };
        }
    }
}

mod tests {
    use crate::lexer::*;
    use insta::assert_debug_snapshot;

    #[test]
    fn single_char_tokens() {
        let source = "+ - * / = . , : ; ! & |";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn double_char_tokens() {
        let source = "+= -= *= /= == != <= >=";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn keyword() {
        let source = "if for while return struct enum union tagunion type";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn idents() {
        let source = "ident _ident ident_123";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn number_literals() {
        let source = "1234 0x9821 0b011 0xfe10 100`000`000_1";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn string_literals() {
        let source = "\"abc\" 'abc' \"abc'\" 'abc\"'";
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn string_literal_escaping() {
        let source: String = ['"', '\\', '\"', '"'].iter().collect();
        let lexer = Lexer::new(&source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }
}
