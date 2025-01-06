mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "if (false) {if (true) {let a = 1;} b = 1;} else { let b = 3;}";
    let mut parser = Parser::new(&source);
    let r = parser.stmt()?;
    println!("{r}");
    Ok(())
}
