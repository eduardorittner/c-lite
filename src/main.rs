mod lexer;

mod parser;
use miette::Result;
use parser::Parser;

fn main() -> Result<()> {
    let source = "0b0c";
    let mut parser = Parser::new(&source);
    let r = parser.parse()?;
    println!("{r:?}");
    Ok(())
}
