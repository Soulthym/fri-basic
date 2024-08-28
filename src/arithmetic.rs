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

type Int = BigInt<1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_field() {
        let g = Fq::from(FqConf::GENERATOR).pow::<Int>(Int::new([3 * 2_u64.pow(20)]));
        g.println_named("g");
        assert!(g.is_order(1024));
    }

    #[test]
    pub fn stark_101_1() {
        let mut a = vec![Fq::from(1), Fq::from(3141592)];
        println!("{:?}", a);
        for l in 2..1023 {
            a.push(a[l - 2] * a[l - 2] + a[l - 1] * a[l - 1]);
        }
        println!("{:?}", a);
        println!("{:?}", a.len());
        assert_eq!(
            a.len(),
            1023,
            "The trace must consist of exactly 1023 elements."
        );
        assert_eq!(
            a[0],
            Fq::from(1),
            "The first element in the trace must be the unit element."
        );
        for i in 2..1023 {
            assert_eq!(
                a[i],
                a[i - 1] * a[i - 1] + a[i - 2] * a[i - 2],
                "The FibonacciSq recursion rule does not apply for index {i}"
            );
        }
        assert_eq!(a[1022], Fq::from(2338775057_u64), "Wrong last element!");
        println!("Success!");
    }
}
