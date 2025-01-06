mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "let a: int = b + 2 * 3 + 1;";
    let mut parser = Parser::new(&source);
    let r = parser.stmt()?;
    println!("{r}");
    Ok(())
}
