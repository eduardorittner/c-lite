mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "struct a { b : void, }";
    let mut parser = Parser::new(&source);
    let r = parser.decl()?;
    println!("{r}");
    Ok(())
}
