use std::any::Any;
use std::cmp::max;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Rem, Sub};

use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use ark_ff::{BigInt, Field};
use rand::prelude::random;

trait Rand<T> {
    fn rand() -> Self;
    fn rand_n(n: u64) -> T;
}

trait Repr {
    fn repr(&self) -> String;
}

trait Named: Repr {
    fn named(&self, name: &str) -> String {
        format!("{name} = {}", self.repr())
    }
}
trait AsName: Named {
    fn as_name(&self, name: &'static str) -> Self
    where
        Self: Any + Sized + Clone,
    {
        let mut new = self.clone();
        new.set_name(name);
        new
    }
    fn set_name(&mut self, name: &'static str) {}
    fn get_name(&self) -> &str {
        ""
    }
}

trait StarkField {
    fn is_order(&self, n: u64) -> bool;
}

trait Variable<T> {
    fn x() -> Self;
    fn eval(x: T) -> T;
}

type Int = BigInt<1>;

#[derive(MontConfig)]
#[modulus = "3221225473"]
#[generator = "5"]
pub struct FqConf;
pub type Fq = Fp64<MontBackend<FqConf, 1>>;

impl Rand<Vec<Fq>> for Fq {
    fn rand() -> Self {
        Fq::from(random::<u64>())
    }
    fn rand_n(n: u64) -> Vec<Fq> {
        let mut res = vec![];
        for _ in 1..n {
            res.push(Fq::rand());
        }
        res
    }
}

impl StarkField for Fq {
    fn is_order(&self, n: u64) -> bool {
        assert!(n >= 1);
        let mut h = Fq::from(1);
        for _ in 1..n {
            h *= self;
            if h == Fq::from(1) {
                return false;
            }
        }
        let next = h * self;
        next == Fq::from(1)
    }
}

impl Repr for Fq {
    fn repr(&self) -> String {
        format!("{self} % {}", FqConf::MODULUS)
    }
}

impl Named for Fq {
    fn named(&self, name: &str) -> String {
        format!("{name} = {}", self.repr())
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone)]
struct Polynomial<F> {
    coefs: Vec<F>,
    name: &'static str,
}

type Poly = Polynomial<Fq>;

impl Poly {
    fn new() -> Self {
        Poly {
            coefs: vec![],
            name: "",
        }
    }
}

impl Repr for Poly {
    fn repr(&self) -> String {
        format!("{} % {}", self, FqConf::MODULUS)
    }
}

impl Named for Poly {
    fn named(&self, name: &str) -> String {
        format!("{name}(x) = {}", self.repr())
    }
}

impl AsName for Poly {
    fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }
    fn get_name(&self) -> &str {
        self.name
    }
}

impl Rand<Poly> for Poly {
    fn rand() -> Self {
        Poly::from(vec![Fq::rand()])
    }
    fn rand_n(n: u64) -> Poly {
        Poly::from(Fq::rand_n(n))
    }
}

impl Variable<Fq> for Poly {
    fn x() -> Self {
        Poly::from(vec![Fq::from(0), Fq::from(1)])
    }
    fn eval(x: Fq) -> Fq {
        unimplemented!()
    }
}

impl From<u64> for Poly {
    fn from(value: u64) -> Self {
        let mut poly = Poly::new();
        poly.coefs.push(Fq::from(value));
        poly
    }
}

impl From<Vec<Fq>> for Poly {
    fn from(coefs: Vec<Fq>) -> Self {
        let mut poly = Poly::new();
        poly.coefs.extend(coefs);
        poly
    }
}

impl Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (expo, coef) in self.coefs.iter().enumerate().rev() {
            if !first {
                write!(f, " + ")?;
            }
            if expo == 0 {
                write!(f, "{}", coef)?;
                return Ok(());
            }
            let a = match coef.0 .0[0] {
                1 => "",
                // -1 => "-", impossible
                _ => &format!("{}", coef),
            };
            if expo == 1 {
                write!(f, "{a}x")?;
            } else {
                write!(f, "{a}x^{expo}")?;
            }
            first = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_field() {
        let g = Fq::from(FqConf::GENERATOR).pow::<Int>(Int::new([3 * 2_u64.pow(20)]));
        println!("{}", g.named("g"));
        assert!(g.is_order(1024));
    }

    #[test]
    pub fn test_poly() {
        let x = Poly::x().as_name("g");
        println!("{}", x);
    }

    #[test]
    pub fn stark_101_1() {
        let p = Poly::from(vec![Fq::from(1), Fq::from(3141592)]).as_name("p");
        println!("{}", p);
        let mut coefs = p.coefs;
        for l in 2..1023 {
            coefs.push(coefs[l - 2] * coefs[l - 2] + coefs[l - 1] * coefs[l - 1]);
        }
        println!("len = {:?}", coefs.len());
        assert_eq!(
            coefs.len(),
            1023,
            "The trace must consist of exactly 1023 elements."
        );
        assert_eq!(
            coefs[0],
            Fq::from(1),
            "The first element in the trace must be the unit element."
        );
        for i in 2..1023 {
            assert_eq!(
                coefs[i],
                coefs[i - 1] * coefs[i - 1] + coefs[i - 2] * coefs[i - 2],
                "The FibonacciSq recursion rule does not apply for index {i}"
            );
        }
        assert_eq!(coefs[1022], Fq::from(2338775057_u64), "Wrong last element!");
        let p = Poly::from(coefs).as_name("p");
        println!("{}", p);
    }
}
