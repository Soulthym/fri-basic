mod arithmetic;
use arithmetic::*;

type P = Poly<FE>;

fn main() {
    let p = P::from(vec![1, 2, 3]);
    println!("{}", p);
    let top = P::from(vec![2, MODULUS - 3, 1]);
    println!("top = {top}");
    let bot = P::x() - P::from(2);
    println!("bot = {bot}");
    let (q, r) = top.divmod(bot.clone());
    println!("{}\n/\n{}\n=\n{}\n(+ {})", top, bot, q, r);
}
