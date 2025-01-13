#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub token: String,
    pub offset: usize,
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
    Tilde,
    Question,
    Xor,
    XorEq,

    // Comparison Operators
    EqEq,
    NotEq,
    AndAnd,
    OrOr,
    GreaterEq,
    Greater,
    SmallerEq,
    Smaller,

    PlusPlus,
    MinusMinus,
    GreaterGreater,
    SmallerSmaller,

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
    Arrow,

    // boolean
    True,
    False,

    // Control-flow
    For,
    If,
    Else,
    Return,
    While,
    Let,

    // Data types
    Enum,
    Struct,
    Union,
    TagUnion,
    Type,
    Fn,

    // Basic Types
    Void,
    Char,
    Int,
    UInt,
    Float,
    Bool,

    Eof,
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::Ident => write!(f, "'identifier'"),
            TokenKind::Literal(kind) => match kind {
                LiteralKind::Hex => write!(f, "'hex'"),
                LiteralKind::Decimal => write!(f, "'decimal'"),
                LiteralKind::Binary => write!(f, "'binary'"),
                LiteralKind::String => write!(f, "'string'"),
            },
            TokenKind::Whitespace => write!(f, "Whitespace"),
            TokenKind::Plus => write!(f, "'+'"),
            TokenKind::PlusEq => write!(f, "'+='"),
            TokenKind::Minus => write!(f, "'-'"),
            TokenKind::MinusEq => write!(f, "'-='"),
            TokenKind::Slash => write!(f, "'/'"),
            TokenKind::SlashEq => write!(f, "'/='"),
            TokenKind::Star => write!(f, "'*'"),
            TokenKind::StarEq => write!(f, "'*='"),
            TokenKind::Percent => write!(f, "'%'"),
            TokenKind::Eq => write!(f, "'='"),
            TokenKind::Not => write!(f, "'!'"),
            TokenKind::And => write!(f, "'&'"),
            TokenKind::Or => write!(f, "'|'"),
            TokenKind::Tilde => write!(f, "'~'"),
            TokenKind::Question => write!(f, "'?'"),
            TokenKind::Xor => write!(f, "'^'"),
            TokenKind::XorEq => write!(f, "'^='"),
            TokenKind::EqEq => write!(f, "'=='"),
            TokenKind::NotEq => write!(f, "'!='"),
            TokenKind::AndAnd => write!(f, "'&&'"),
            TokenKind::OrOr => write!(f, "'||'"),
            TokenKind::GreaterEq => write!(f, "'>='"),
            TokenKind::Greater => write!(f, "'>'"),
            TokenKind::SmallerEq => write!(f, "'<='"),
            TokenKind::Smaller => write!(f, "'<'"),
            TokenKind::PlusPlus => write!(f, "'++'"),
            TokenKind::MinusMinus => write!(f, "'--'"),
            TokenKind::GreaterGreater => write!(f, "'>>'"),
            TokenKind::SmallerSmaller => write!(f, "'<<'"),
            TokenKind::OpenParen => write!(f, "'('"),
            TokenKind::CloseParen => write!(f, "')'"),
            TokenKind::OpenBracket => write!(f, "'{{'"),
            TokenKind::CloseBracket => write!(f, "'}}'"),
            TokenKind::OpenBrace => write!(f, "'['"),
            TokenKind::CloseBrace => write!(f, "']'"),
            TokenKind::Comma => write!(f, "','"),
            TokenKind::Dot => write!(f, "'.'"),
            TokenKind::Colon => write!(f, "':'"),
            TokenKind::Semicolon => write!(f, "';'"),
            TokenKind::Arrow => write!(f, "'->'"),
            TokenKind::True => write!(f, "'true'"),
            TokenKind::False => write!(f, "'false'"),
            TokenKind::For => write!(f, "'for'"),
            TokenKind::If => write!(f, "'if'"),
            TokenKind::Else => write!(f, "'else'"),
            TokenKind::Return => write!(f, "'return'"),
            TokenKind::While => write!(f, "'while'"),
            TokenKind::Enum => write!(f, "'enum'"),
            TokenKind::Struct => write!(f, "'strucT'"),
            TokenKind::Union => write!(f, "'union'"),
            TokenKind::TagUnion => write!(f, "'tagUnion'"),
            TokenKind::Type => write!(f, "'type'"),
            TokenKind::Fn => write!(f, "'fn'"),
            TokenKind::Eof => write!(f, "'eof'"),
            TokenKind::Void => write!(f, "'void'"),
            TokenKind::Char => write!(f, "'char'"),
            TokenKind::Int => write!(f, "'int'"),
            TokenKind::UInt => write!(f, "'uint'"),
            TokenKind::Float => write!(f, "'float'"),
            TokenKind::Bool => write!(f, "'bool'"),
            TokenKind::Let => write!(f, "'let'"),
        }
    }
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

impl Iterator for Lexer<'_> {
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
                ThreePossibilitites(char), // TODO naming????
                                           // relates to characters which have more than 2 variations
                                           // e.g. '+' can be '+', '+=' or '++'
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
                '?' => return simple_token(Question),
                '~' => return simple_token(Tilde),
                '+' => State::ThreePossibilitites('+'),
                '-' => State::ThreePossibilitites('-'),
                '*' => State::IfEqElse('=', StarEq, Star),
                '/' => State::IfEqElse('=', SlashEq, Slash),
                '=' => State::IfEqElse('=', EqEq, Eq),
                '!' => State::IfEqElse('=', NotEq, Not),
                '>' => State::ThreePossibilitites('>'),
                '<' => State::ThreePossibilitites('<'),
                '&' => State::IfEqElse('&', AndAnd, And),
                '|' => State::IfEqElse('|', OrOr, Or),
                '^' => State::IfEqElse('=', XorEq, Xor),
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
                    self.offset += whitespace_offset;
                    self.rest = &self.rest[whitespace_offset..];
                    continue;
                }
                State::Ident => {
                    let first_non_ident = char_onwards
                        .find(|c| !matches!(c,'a'..='z'| 'A'..='Z'| '_'| '0'..='9'))
                        .unwrap_or(char_onwards.len());

                    let lit = &char_onwards[..first_non_ident];
                    let lit_offset = lit.len() - char.len_utf8();

                    let kind = match lit {
                        "let" => Let,
                        "for" => For,
                        "if" => If,
                        "else" => Else,
                        "return" => Return,
                        "while" => While,
                        "enum" => Enum,
                        "struct" => Struct,
                        "union" => Union,
                        "tagunion" => TagUnion,
                        "type" => Type,
                        "fn" => Fn,
                        "true" => True,
                        "false" => False,
                        "void" => Void,
                        "char" => Char,
                        "int" => Int,
                        "uint" => UInt,
                        "bool" => Bool,
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
                State::ThreePossibilitites(char) => {
                    if self.rest.starts_with('=') {
                        let token = char_onwards[..char_str.len() + char.len_utf8()].to_string();
                        self.offset += char.len_utf8();
                        self.rest = &self.rest[char.len_utf8()..];
                        match char {
                            '+' => {
                                return Some(Token {
                                    kind: TokenKind::PlusEq,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '-' => {
                                return Some(Token {
                                    kind: TokenKind::MinusEq,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '>' => {
                                return Some(Token {
                                    kind: TokenKind::GreaterEq,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '<' => {
                                return Some(Token {
                                    kind: TokenKind::SmallerEq,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            _ => unreachable!(),
                        }
                    } else if self.rest.starts_with(char) {
                        let token = char_onwards[..char_str.len() + char.len_utf8()].to_string();
                        self.offset += char.len_utf8();
                        self.rest = &self.rest[char.len_utf8()..];
                        match char {
                            '+' => {
                                return Some(Token {
                                    kind: TokenKind::PlusPlus,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '-' => {
                                return Some(Token {
                                    kind: TokenKind::MinusMinus,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '>' => {
                                return Some(Token {
                                    kind: TokenKind::GreaterGreater,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            '<' => {
                                return Some(Token {
                                    kind: TokenKind::SmallerSmaller,
                                    token,
                                    offset: char_offset,
                                })
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        match char {
                            '+' => {
                                return Some(Token {
                                    kind: TokenKind::Plus,
                                    token: char_str.to_string(),
                                    offset: char_offset,
                                })
                            }
                            '-' => {
                                if self.rest.starts_with(">") {
                                    let token =
                                        char_onwards[..char_str.len() + '>'.len_utf8()].to_string();
                                    self.offset += '>'.len_utf8();
                                    self.rest = &self.rest['>'.len_utf8()..];
                                    return Some(Token {
                                        kind: TokenKind::Arrow,
                                        token,
                                        offset: char_offset,
                                    });
                                };
                                return Some(Token {
                                    kind: TokenKind::Minus,
                                    token: char_str.to_string(),
                                    offset: char_offset,
                                });
                            }
                            '>' => {
                                return Some(Token {
                                    kind: TokenKind::Greater,
                                    token: char_str.to_string(),
                                    offset: char_offset,
                                })
                            }
                            '<' => {
                                return Some(Token {
                                    kind: TokenKind::Smaller,
                                    token: char_str.to_string(),
                                    offset: char_offset,
                                })
                            }
                            _ => unreachable!(),
                        }
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
                                && char_onwards.chars().nth(i - 1).unwrap() != '\\'
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

#[cfg(test)]
mod tests {
    use crate::lexer::*;
    use insta::assert_debug_snapshot;

    #[test]
    fn single_char_tokens() {
        let source = "+ - * / = . , : ; ! ~ & | % [ ]";
        let lexer = Lexer::new(source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn double_char_tokens() {
        let source = "+= ++ -= -- *= /= == != <= >= << >> ->";
        let lexer = Lexer::new(source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn keyword() {
        let source = "if for while return struct enum union tagunion type";
        let lexer = Lexer::new(source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn idents() {
        let source = "ident _ident ident_123";
        let lexer = Lexer::new(source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn number_literals() {
        let source = "1234 0x9821 0b011 0xfe10 100`000`000_1";
        let lexer = Lexer::new(source);
        let result: Vec<Token> = lexer.into_iter().collect();
        assert_debug_snapshot!(result);
    }

    #[test]
    fn string_literals() {
        let source = "\"abc\" 'abc' \"abc'\" 'abc\"'";
        let lexer = Lexer::new(source);
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

    #[test]
    #[should_panic]
    fn invalid_char() {
        let source = "🚫";
        let lexer = Lexer::new(source);
        let _: Vec<Token> = lexer.into_iter().collect();
    }

    #[test]
    fn test_display() {
        let source = "ident 0x1f 0b10 23 \"string\" + += - -= / /= * *= % = ! & ^ ~ | ? ^= == != && ^^ <= || < >= > ++ -- >> << ( ) [ ] { } , . : ; -> true false for if else return while let enum struct union tagunion type void char int uint float bool)";
        let lexer = Lexer::new(source);
        let result: Vec<String> = lexer.into_iter().map(|t| t.kind.to_string()).collect();
        assert_debug_snapshot!(result);
    }
}
