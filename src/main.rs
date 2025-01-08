mod lexer;
pub use lexer::Lexer;

mod parser;
use miette::Result;
pub use parser::Parser;

fn main() -> Result<()> {
    let source = "fn a(b: void, c: mytpe) -> shitty {let a = b;} fn b() {let a = 2;} let a = 1;";
    let mut parser = Parser::new(&source);
    let r = parser.parse()?;
    for a in r {
        println!("{a}")
    }
    Ok(())
}
