use std::any::Any;
use std::cmp::max;
use std::fmt::Display;
use std::iter::zip;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use ark_ff::{BigInt, Field, One, PrimeField, Zero};
use rand::prelude::random;

trait Rand {
    type Multi;
    fn rand() -> Self;
    fn rand_n(n: u64) -> Self::Multi;
}
trait RandExc<T> {
    type Multi;
    fn rand_except(excluded: &Vec<T>) -> Self;
    fn rand_n_except(n: u64, excluded: &Vec<T>) -> Self::Multi;
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
    fn set_name(&mut self, name: &'static str);
    fn get_name(&self) -> &str;
}

trait IsOrder {
    fn is_order(&self, n: u64) -> bool;
}

trait UniVariate {
    fn x() -> Self;
    fn eval(self, x: Fq) -> Fq;
    fn monomial(degree: usize, coef: Fq) -> Self;
    fn gen_linear_term(value: Fq) -> Self;
    fn degree(&mut self) -> usize;
    fn get_coef(&self, degree: usize) -> Fq;
    fn compose(self, other: Self) -> Self;
}

trait VecUtils {
    fn zip_op(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self;
    fn ziplongest_map(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self;
    fn cross_map(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self;
    fn trimtrailingzeros(&mut self);
}

trait DivMod<T> {
    fn divmod(self, other: T) -> (Self, Self)
    where
        Self: Sized;
    fn div(self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).0
    }
    fn modulus(self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).1
    }
}

trait Pow {
    type Rhs;
    fn pow(self, rhs: Self::Rhs) -> Self;
}

type Int = BigInt<1>;

#[derive(MontConfig)]
#[modulus = "3221225473"]
#[generator = "5"]
pub struct FqConf;
pub type Fq = Fp64<MontBackend<FqConf, 1>>;

impl Rand for Fq {
    type Multi = Vec<Self>;
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

impl RandExc<Fq> for Fq {
    type Multi = Vec<Fq>;
    fn rand_except(excluded: &Vec<Fq>) -> Self {
        let mut res = Fq::rand();
        while excluded.iter().any(|&excl| res == excl) {
            res = Fq::rand();
        }
        res
    }
    fn rand_n_except(n: u64, excluded: &Vec<Fq>) -> Self::Multi {
        let mut res = vec![];
        for _ in 1..n {
            res.push(Fq::rand_except(excluded));
        }
        res
    }
}

impl IsOrder for Fq {
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

    fn zeros(n: usize) -> Self {
        let mut res = Poly::new();
        for _ in 0..n {
            res.coefs.push(Fq::zero());
        }
        res
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

impl Rand for Poly {
    type Multi = Poly;
    fn rand() -> Self {
        Poly::from(Fq::rand())
    }
    fn rand_n(n: u64) -> Poly {
        Poly::from(Fq::rand_n(n))
    }
}

impl RandExc<Fq> for Poly {
    type Multi = Self;
    fn rand_except(excluded: &Vec<Fq>) -> Self {
        Poly::from(Fq::rand_except(excluded))
    }
    fn rand_n_except(n: u64, excluded: &Vec<Fq>) -> Self::Multi {
        Poly::from(Fq::rand_n_except(n, excluded))
    }
}

impl RandExc<Poly> for Poly {
    type Multi = Self;
    fn rand_except(excluded: &Vec<Poly>) -> Self {
        let mut res = Poly::rand();
        while excluded.iter().any(|excl| res == *excl) {
            res = Poly::rand();
        }
        res
    }
    fn rand_n_except(n: u64, excluded: &Vec<Poly>) -> Self::Multi {
        let mut res = Poly::rand_n(n);
        while excluded.iter().any(|excl| res == *excl) {
            res = Poly::rand_n(n);
        }
        res
    }
}

impl UniVariate for Poly {
    fn x() -> Self {
        Poly::from(vec![Fq::from(0), Fq::from(1)])
    }
    fn eval(self, x: Fq) -> Fq {
        let mut res = Fq::from(0);
        for coef in self.coefs.iter().rev() {
            res *= x + coef;
        }
        res
    }
    fn monomial(degree: usize, coef: Fq) -> Self {
        let mut res = Poly::zeros(degree);
        res.coefs.push(coef);
        res
    }
    fn gen_linear_term(value: Fq) -> Self {
        Poly::x() - Poly::from(value)
    }
    fn degree(&mut self) -> usize {
        self.trimtrailingzeros();
        self.coefs.len() - 1
    }
    fn get_coef(&self, degree: usize) -> Fq {
        match self.coefs.get(degree) {
            Some(value) => *value,
            None => Fq::from(0),
        }
    }
    fn compose(self, other: Poly) -> Self {
        let mut res = Poly::new();
        for coef in self.coefs.iter().rev() {
            res = res * other.clone() + Poly::from(*coef);
        }
        res
    }
}

impl VecUtils for Poly {
    fn zip_op(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self {
        let mut res = Poly::new();
        for (left, right) in zip(self.coefs, other.coefs) {
            res.coefs.push(op(left, right));
        }
        res
    }

    fn ziplongest_map(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self {
        let mut res = Poly::new();
        let longest = max(self.coefs.len(), other.coefs.len());
        for degree in 0..longest {
            let left = self.get_coef(degree);
            let right = other.get_coef(degree);
            res.coefs.push(op(left, right));
        }
        res
    }

    fn cross_map(self, other: Poly, op: impl Fn(Fq, Fq) -> Fq) -> Self {
        let mut res = Poly::zeros(self.coefs.len() + other.coefs.len());
        for (lexp, lhs) in self.coefs.iter().enumerate() {
            for (rexp, rhs) in self.coefs.iter().enumerate() {
                res.coefs[lexp + rexp] = op(*lhs, *rhs);
            }
        }
        res
    }

    fn trimtrailingzeros(&mut self) {
        for coef in self.coefs.clone().iter().rev() {
            if *coef != Fq::zero() {
                break;
            }
            let _ = self.coefs.pop();
        }
    }
}

impl Pow for Poly {
    type Rhs = Poly;
    fn pow(self, rhs: Self::Rhs) -> Self {
        let mut other = rhs;
        assert!(other >= Poly::zero());
        let mut res = Poly::one();
        let mut cur = self;
        loop {
            if other.clone() % 2.into() != 0.into() {
                res = res * cur.clone();
            }
            other = other / 2.into();
            if other == 0.into() {
                break;
            }
            cur = cur.clone() * cur.clone();
        }
        res
    }
}

impl DivMod<Poly> for Poly {
    fn divmod(self, other: Poly) -> (Self, Self) {
        assert!(other != Poly::zero(), "Cannot divide by zero");
        let mut lhs: Poly = self.clone();
        lhs.trimtrailingzeros();
        let mut rhs: Poly = other.clone();
        rhs.trimtrailingzeros();
        if lhs.coefs.is_empty() {
            return (Poly::zero(), Poly::zero());
        }
        let mut rem = lhs;
        let (lenl, lenr) = (rem.coefs.len(), other.coefs.len());
        let (mut keep_going, mut deg_diff) = if lenl < lenr {
            (true, lenl - lenr)
        } else {
            (false, lenr - lenl)
        };
        let mut quotient = Poly::zeros(deg_diff + 1);
        let g_msc_inv = &rhs.coefs.last().unwrap().inverse().unwrap();
        while keep_going {
            let tmp = rem.coefs.last().unwrap() * g_msc_inv;
            quotient.coefs[deg_diff] += tmp;
            let mut last_non_zero = deg_diff - 1;
            for (i, coef) in rhs.coefs.iter().enumerate() {
                let i = i + deg_diff;
                rem.coefs[i] -= tmp * coef;
                if rem.coefs[i] != Fq::zero() {
                    last_non_zero = i;
                }
                let mut rem_coefs = vec![];
                for rem_coef in rem.coefs.iter().take(last_non_zero + 1) {
                    rem_coefs.push(*rem_coef)
                }
                rem = Poly::from(rem_coefs);
                (keep_going, deg_diff) = if lenl < lenr {
                    (true, lenl - lenr)
                } else {
                    (false, lenr - lenl)
                };
            }
        }
        quotient.trimtrailingzeros();
        (quotient, rem)
    }
}

impl Add<Poly> for Poly {
    type Output = Self;
    fn add(self, other: Poly) -> Self::Output {
        self.ziplongest_map(other, |a, b| a + b)
    }
}

impl Sub<Poly> for Poly {
    type Output = Self;
    fn sub(self, other: Poly) -> Self {
        self.ziplongest_map(other, |a, b| a - b)
    }
}

impl Neg for Poly {
    type Output = Self;
    fn neg(self) -> Self {
        Poly::from(Fq::MODULUS) * self
    }
}

impl Mul<Poly> for Poly {
    type Output = Self;
    fn mul(self, other: Poly) -> Self {
        self.cross_map(other, |a, b| a * b)
    }
}

impl Div<Poly> for Poly {
    type Output = Self;
    fn div(self, other: Poly) -> Self {
        DivMod::div(self, other)
    }
}

impl Rem<Poly> for Poly {
    type Output = Self;
    fn rem(self, other: Poly) -> Self {
        DivMod::modulus(self, other)
    }
}

impl Zero for Poly {
    fn is_zero(&self) -> bool {
        *self == Poly::zero()
    }
    fn zero() -> Self {
        Poly::new()
    }
}

impl One for Poly {
    fn is_one(&self) -> bool {
        *self == Poly::one()
    }
    fn one() -> Self {
        Poly::from(Fq::ONE)
    }
}

impl From<Vec<u64>> for Poly {
    fn from(coefs: Vec<u64>) -> Self {
        let mut poly = Poly::new();
        for coef in coefs {
            poly.coefs.push(Fq::from(coef));
        }
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

impl From<Fq> for Poly {
    fn from(coef: Fq) -> Self {
        Poly::from(vec![coef])
    }
}

impl From<Int> for Poly {
    fn from(coef: Int) -> Self {
        Poly::from(Fq::from(coef))
    }
}

impl From<u64> for Poly {
    fn from(coef: u64) -> Self {
        Poly::from(Fq::from(coef))
    }
}

impl Display for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (exp, coef) in self.coefs.iter().enumerate().rev() {
            if *coef == Fq::from(0) {
                continue;
            }
            let prefix = if first { "" } else { " + " };
            let rcoef = if *coef != Fq::from(1) {
                &format!("{}", coef)
            } else {
                ""
            };
            match exp {
                0 => write!(f, "{}{}", prefix, coef)?,
                1 => write!(f, "{}{}x", prefix, rcoef,)?,
                _ => write!(f, "{}{}x^{}", prefix, rcoef, exp)?,
            }
            first = false;
        }
        write!(f, " % {}", FqConf::MODULUS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_all() {
        println!("stark_101_1()");
        stark_101_1();
        println!("test_field()");
        test_field();
        println!("test_poly()");
        test_poly();
    }

    pub fn test_field() {
        let g = Fq::from(FqConf::GENERATOR).pow::<Int>(Int::new([3 * 2_u64.pow(20)]));
        println!("{}", g.named("g"));
        assert!(g.is_order(1024));
    }

    pub fn test_poly() {
        let x = Poly::x().as_name("p");
        let one = Poly::one();
        println!("x? {:?}", x);
        println!("{}", x);
        println!("one? {:?}", one);
        println!("{}", one);
        println!("x + 1 = {}", x + 1.into());
        let p1 = Poly::from(Fq::zero() - Fq::from(5)) + Poly::x();
        let p2 = Poly::from(vec![Fq::from(0) - Fq::from(5), Fq::from(1)]);
        println!("p1 = {}", p1);
        println!("p2 = {}", p2);
        assert_eq!(p1, p2);
    }

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
