#![allow(dead_code, unused_imports)]
mod field;
mod fri;
mod poly;
mod traits;
use ark_ff::BigInt;
use field::*;
use fri::*;
use poly::*;
use traits::*;

type P = Poly<FE>;

fn main() {
    let s = FE::from(BigInt!("10"));
    println!("s = {}\n {:?}", s.repr(), s);
    let u = FE::from(10);
    println!("u = {}\n {:?}", u.repr(), u);
    let g = gen();
    println!("g = {}\n {:?}", g.repr(), g);
}
