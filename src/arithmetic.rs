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
    fn rand_except(excluded: &[T]) -> Self;
    fn rand_n_except(n: u64, excluded: &[T]) -> Self::Multi;
}

trait ReprMod {
    fn repr(&self) -> String;
    fn reprmod(&self) -> String {
        format!("{} % {}", self.repr(), FqConf::MODULUS)
    }
}

trait Named: ReprMod {
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

trait VecUtils<F, Rhs = Self> {
    fn zip_op(self, other: Rhs, op: impl Fn(F, F) -> F) -> Self;
    fn ziplongest_map(self, other: Rhs, op: impl Fn(F, F) -> F) -> Self;
    fn cross_map(self, other: Rhs, op: impl Fn(F, F) -> F) -> Self;
    fn trimtrailingzeros(&mut self);
}

trait Algebra<T, O> {
    fn add(self, other: T) -> O;
    fn neg(self) -> O;
    fn sub(self, other: T) -> O;
    fn mul(self, other: T) -> O;
    fn divmod(self, other: T) -> (O, O)
    where
        Self: Sized;
    fn div(self, other: T) -> O
    where
        Self: Sized,
    {
        self.divmod(other).0
    }
    fn modulus(self, other: T) -> O
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
        let res = Fq::from(random::<u64>());
        println!("{res}");
        res
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
    fn rand_except(excluded: &[Fq]) -> Self {
        let mut res = Fq::rand();
        while excluded.iter().any(|&excl| res == excl) {
            res = Fq::rand();
        }
        res
    }
    fn rand_n_except(n: u64, excluded: &[Fq]) -> Self::Multi {
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

impl ReprMod for Fq {
    fn repr(&self) -> String {
        let m = Fq::MODULUS.0[0] as i128;
        let h = m / 2;
        let rep = (self.0 .0[0] as i128 + h) % m - h;
        format!("{}", rep)
    }
}

impl Named for Fq {
    fn named(&self, name: &str) -> String {
        format!("{name} = {}", self.reprmod())
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

impl<T> ReprMod for Polynomial<T> {
    fn repr(&self) -> String {
        format!("{}", self)
    }
}

impl<T> Named for Polynomial<T> {
    fn named(&self, name: &str) -> String {
        format!("{name}(x) = {}", self.reprmod())
    }
}

impl<T> AsName for Polynomial<T> {
    fn set_name(&mut self, name: &'static str) {
        self.name = name;
    }
    fn get_name(&self) -> &str {
        self.name
    }
}

impl<F> Rand for Polynomial<F>
where
    F: Zero + Rand + RandExc<F>,
    Polynomial<F>: From<F>,
{
    type Multi = Polynomial<F>;
    fn rand() -> Self {
        Polynomial::from(F::rand())
    }
    fn rand_n(n: u64) -> Polynomial<F> {
        let leading = F::rand_except(&[F::zero()]);
        let mut res = Polynomial::from(F::rand_n(n));
        res.coefs.push(leading);
        res
    }
}

impl RandExc<Fq> for Poly {
    type Multi = Self;
    fn rand_except(excluded: &[Fq]) -> Self {
        Poly::from(Fq::rand_except(excluded))
    }
    fn rand_n_except(n: u64, excluded: &[Fq]) -> Self::Multi {
        Poly::from(Fq::rand_n_except(n, excluded))
    }
}

impl RandExc<Poly> for Poly {
    type Multi = Self;
    fn rand_except(excluded: &[Poly]) -> Self {
        let mut res = Poly::rand();
        while excluded.iter().any(|excl| res == *excl) {
            res = Poly::rand();
        }
        res
    }
    fn rand_n_except(n: u64, excluded: &[Poly]) -> Self::Multi {
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

impl<F> VecUtils<F> for Polynomial<F> {
    fn zip_op(self, other: Polynomial<F>, op: impl Fn(F, F) -> F) -> Self {
        let mut res = Polynomial::new();
        for (left, right) in zip(self.coefs, other.coefs) {
            res.coefs.push(op(left, right));
        }
        res
    }

    fn ziplongest_map(self, other: Polynomial<F>, op: impl Fn(F, F) -> F) -> Self {
        let mut res = Polynomial::new();
        let longest = max(self.coefs.len(), other.coefs.len());
        for degree in 0..longest {
            let left = self.get_coef(degree);
            let right = other.get_coef(degree);
            res.coefs.push(op(left, right));
        }
        res
    }

    fn cross_map(self, other: Polynomial<F>, op: impl Fn(F, F) -> F) -> Self {
        let mut res = Polynomial::zeros(self.coefs.len() + other.coefs.len());
        for (lexp, lhs) in self.coefs.iter().enumerate() {
            for (rexp, rhs) in self.coefs.iter().enumerate() {
                res.coefs[lexp + rexp] = op(*lhs, *rhs);
            }
        }
        res
    }

    fn trimtrailingzeros(&mut self) {
        for coef in self.coefs.clone().iter().rev() {
            if *coef != F::zero() {
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
            if other.clone() % Poly::from(2) != Poly::zero() {
                res = res * cur.clone();
            }
            other = other / Poly::from(2);
            if other == 0.into() {
                break;
            }
            cur = cur.clone() * cur.clone();
        }
        res
    }
}

impl<T, F> Algebra<T, Polynomial<F>> for Polynomial<F>
where
    Polynomial<F>: From<T> + VecUtils<F> + Algebra<Polynomial<F>, Polynomial<F>>,
    T: Clone + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T>,
{
    fn add(self, other: T) -> Polynomial<F> {
        self.ziplongest_map(Polynomial::from(other), |a, b| a + b)
    }

    fn neg(self) -> Polynomial<F> {
        Polynomial::from(F::MODULUS) * self
    }

    fn sub(self, other: T) -> Polynomial<F> {
        self.ziplongest_map(Polynomial::from(other), |a, b| a - b)
    }

    fn mul(self, other: T) -> Polynomial<F> {
        self.cross_map(Polynomial::from(other), |a, b| a * b)
    }

    fn divmod(self, other: T) -> (Polynomial<F>, Polynomial<F>) {
        println!("DIVMOD:\n  self = {self}\n  other = {other}");
        assert!(
            Polynomial::from(other) != Polynomial::zero(),
            "Cannot divide by zero"
        );
        let mut lhs: Polynomial<F> = self.clone();
        lhs.trimtrailingzeros();
        println!("lhs = {lhs}");
        let mut rhs: Polynomial<F> = other.clone();
        rhs.trimtrailingzeros();
        println!("rhs = {rhs}");
        if lhs.coefs.is_empty() {
            return (Polynomial::zero(), Polynomial::zero());
        }
        let mut rem = lhs;
        let (lenl, lenr) = (rem.coefs.len(), other.coefs.len());
        println!("lenl = {lenl}");
        println!("lenr = {lenr}");
        let (mut keep_going, mut deg_diff) = if lenl < lenr {
            (false, lenr - lenl)
        } else {
            (true, lenl - lenr)
        };
        println!("deg_diff = {deg_diff}");
        println!("keep_going = {keep_going}");
        let mut quotient = Polynomial::zeros(deg_diff + 1);
        println!("quotient = {quotient}");
        let g_msc_inv = &rhs.coefs.last().unwrap().inverse().unwrap();
        println!("g_msc_inv = {g_msc_inv}");
        while keep_going {
            println!("  while keep_going = {keep_going}");
            let tmp = rem.coefs.last().unwrap() * g_msc_inv;
            println!("    tmp = {tmp}");
            quotient.coefs[deg_diff] += tmp;
            println!("    quotient = {quotient}");
            let mut last_non_zero = deg_diff - 1;
            println!("    last_non_zero = {last_non_zero}");
            for (i, coef) in rhs.coefs.iter().enumerate() {
                println!("      for i = {i}");
                println!("        coef = {coef}");
                let i = i + deg_diff;
                println!("        i = {i}");
                rem.coefs[i] -= tmp * coef;
                println!("        rem = {rem}");
                if rem.coefs[i] != F::zero() {
                    last_non_zero = i;
                    println!("        last_non_zero = {last_non_zero}");
                }
                let mut rem_coefs = vec![];
                for rem_coef in rem.coefs.iter().take(last_non_zero + 1) {
                    rem_coefs.push(*rem_coef)
                }
                rem = Polynomial::from(rem_coefs);
                println!("        rem = {rem}");
                (keep_going, deg_diff) = if lenl < lenr {
                    (false, lenr - lenl)
                } else {
                    (true, lenl - lenr)
                };
                println!("        keep_going = {keep_going}");
                println!("        deg_diff = {deg_diff}");
            }
        }
        quotient.trimtrailingzeros();
        println!("rem = {rem}");
        println!("quotient = {quotient}");
        (quotient, rem)
    }
}

impl<T, Rhs> Add<Rhs> for Polynomial<T>
where
    Polynomial<T>: From<Rhs> + Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn add(self, rhs: Rhs) -> Self::Output {
        Algebra::add(self, Self::from(rhs))
    }
}

impl<T> Neg for Polynomial<T>
where
    Polynomial<T>: Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Algebra::neg(self)
    }
}

impl<T, Rhs> Sub<Rhs> for Polynomial<T>
where
    Polynomial<T>: From<Rhs> + Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn sub(self, rhs: Rhs) -> Self::Output {
        Algebra::sub(self, Self::from(rhs))
    }
}

impl<T, Rhs> Mul<Rhs> for Polynomial<T>
where
    Polynomial<T>: From<Rhs> + Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn mul(self, rhs: Rhs) -> Self::Output {
        Algebra::mul(self, Self::from(rhs))
    }
}

impl<T, Rhs> Div<Rhs> for Polynomial<T>
where
    Polynomial<T>: From<Rhs> + Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn div(self, rhs: Rhs) -> Self::Output {
        Algebra::div(self, Self::from(rhs))
    }
}

impl<T, Rhs> Rem<Rhs> for Polynomial<T>
where
    Polynomial<T>: From<Rhs> + Algebra<Polynomial<T>, Polynomial<T>>,
{
    type Output = Self;
    fn rem(self, rhs: Rhs) -> Self::Output {
        Algebra::modulus(self, Self::from(rhs))
    }
}

impl<T> Zero for Polynomial<T> {
    fn is_zero(&self) -> bool {
        *self == Polynomial::zero()
    }
    fn zero() -> Self {
        Polynomial::new()
    }
}

impl<F> One for Polynomial<F> {
    fn is_one(&self) -> bool {
        *self == Polynomial::one()
    }
    fn one() -> Self {
        Polynomial::from(F::ONE)
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
            let rep = coef.repr();
            let mut rcoef = rep.as_str();
            if rcoef == "0" {
                continue;
            }
            let prefix = if first {
                ""
            } else if rcoef.chars().nth(0).unwrap() == '-' {
                rcoef = &rcoef[1..rcoef.len() - 1];
                " - "
            } else {
                " + "
            };
            let rcoef = if rcoef == "1" { "" } else { rcoef };
            match exp {
                0 => write!(f, "{}{}", prefix, rcoef)?,
                1 => write!(f, "{}{}x", prefix, rcoef,)?,
                _ => write!(f, "{}{}x^{}", prefix, rcoef, exp)?,
            }
            first = false;
        }
        if first {
            write!(f, "0")?;
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
        println!("test_div_random_poly()");
        test_div_random_poly();
    }

    pub fn test_field() {
        let g = Fq::from(FqConf::GENERATOR).pow::<Int>(Int::new([3 * 2_u64.pow(20)]));
        println!("{}", g.named("g"));
        assert!(g.is_order(1024));
    }

    pub fn test_poly() {
        type P = Poly;
        let mut x = P::x().as_name("p");
        let one = P::one();
        println!("x? {:?}", x);
        println!("{}", x);
        println!("one? {:?}", one);
        println!("{}", one);
        println!("x + 1 = {}", x + one);
        let p1 = P::from(Fq::zero() - Fq::from(5)) + P::x();
        let p2 = P::from(vec![Fq::from(0) - Fq::from(5), Fq::from(1)]);
        println!("p1 = {}", p1);
        println!("p2 = {}", p2);
        assert_eq!(p1, p2);
        println!("x^2 + 1 = {}", x ^ 2 + one);
    }

    pub fn test_div_random_poly() {
        for _ in 0..20 {
            let deg_a = random::<u64>() % 5;
            let deg_b = random::<u64>() % 5;
            println!("deg_a = {deg_a}");
            println!("deg_b = {deg_b}");
            let a = Poly::rand_n(deg_a);
            let mut b = Poly::rand_n(deg_b);
            println!("a = {a}");
            println!("b = {b}");
            let (q, mut r) = a.clone().divmod(b.clone());
            let d = r.clone() + q * b.clone();
            println!("assert!(r.degree() < b.degree())");
            assert!(r.degree() < b.degree());
            println!("assert!(d == a)");
            assert!(d == a);
        }
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
