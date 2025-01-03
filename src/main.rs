mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "let a: int = 1";
    let mut parser = Parser::new(&source);
    let r = parser.decl()?;
    println!("{r}");
    Ok(())
}
