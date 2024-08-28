use ark_ff::BigInt;
use std::cmp::max;
use std::ops::{Add, Div, Mul, Rem, Sub};

use ark_ff::fields::{Field, Fp64, MontBackend, MontConfig, PrimeField};
use rand::prelude::random;

trait Print {
    fn print(&self);
    fn println(&self);
    fn print_named(&self, name: &str);
    fn println_named(&self, name: &str);
}
#[derive(MontConfig)]
#[modulus = "3221225473"]
#[generator = "5"]
pub struct FqConf;
pub type Fq = Fp64<MontBackend<FqConf, 1>>;

trait IsOrder {
    fn is_order(&self, n: u64) -> bool;
}

impl IsOrder for Fq {
    fn is_order(&self, n: u64) -> bool {
        assert!(n >= 1);
        let mut h = Fq::from(1);
        for i in 1..n {
            h *= self;
            if h == Fq::from(1) {
                return false;
            }
        }
        let next = h * self;
        next == Fq::from(1)
    }
}

impl Print for Fq {
    fn println(&self) {
        self.print();
        println!();
    }
    fn print(&self) {
        print!("{} % {}", self, FqConf::MODULUS);
    }
    fn println_named(&self, name: &str) {
        self.print_named(name);
        println!();
    }
    fn print_named(&self, name: &str) {
        print!("{}(x) = ", name);
        self.print();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    pub fn test_field() {
        let g = Fq::from(FqConf::GENERATOR).pow::<BigInt<1>>(BigInt::<1>::new([3 * 2_u64.pow(20)]));
        g.println_named("g");
        assert!(g.is_order(1024));
    }
}
