mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "a = *p++";
    let mut parser = Parser::new(&source);
    let r = parser.parse()?;
    println!("{r}");
    Ok(())
}
