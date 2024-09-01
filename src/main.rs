mod arithmetic;
use arithmetic::*;

type P = Poly<FE>;

fn main() {
    let p = P::from(vec![1, 2, 3]);
    println!("{}", p);
    let x = P::x();
    println!("x = {}", x);
}
