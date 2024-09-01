#![allow(dead_code)]
use lambdaworks_math::field::fields::u64_prime_field::{F17, FE17};
use rand::random;
use std::cmp::max;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

pub trait Repr {
    fn repr(&self) -> String;
    fn dbg(&self) -> String {
        self.repr()
    }
}

pub trait ReprMod: Repr {
    fn repr_mod(&self) -> String {
        format!("{} % {}", self.repr(), MODULUS)
    }
}

pub trait Rand {
    type Multi;
    fn rand() -> Self;
    fn rand_n(n: u64) -> Self::Multi;
}

pub trait RandExc<T> {
    type Multi;
    fn rand_except(excluded: &Vec<T>) -> Self;
    fn rand_n_except(n: u64, excluded: &Vec<T>) -> Self::Multi;
}

pub trait DivMod<T> {
    fn divmod(&self, other: T) -> (Self, Self)
    where
        Self: Sized;
    fn div(&self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).0
    }
    fn modulus(&self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).1
    }
}

//pub type F = Stark252PrimeField;
//pub type FE = FieldElement<F>;
pub type F = F17;
pub type FE = FE17;
pub const MODULUS: u64 = 17;

impl Rand for FE {
    type Multi = Vec<Self>;
    fn rand() -> Self {
        FE::from(random::<u64>())
    }
    fn rand_n(n: u64) -> Vec<FE> {
        let mut res = vec![];
        for _ in 0..n {
            res.push(FE::rand());
        }
        res
    }
}

impl RandExc<FE> for FE {
    type Multi = Vec<FE>;
    fn rand_except(excluded: &Vec<FE>) -> Self {
        let mut res = FE::rand();
        while excluded.iter().any(|&excl| res == excl) {
            res = FE::rand();
        }
        res
    }
    fn rand_n_except(n: u64, excluded: &Vec<FE>) -> Self::Multi {
        let mut res = vec![];
        for _ in 0..n {
            res.push(FE::rand_except(excluded));
        }
        res
    }
}

impl Repr for FE {
    fn repr(&self) -> String {
        let m = MODULUS as i128;
        let h = m / 2;
        let v: i128 = (*self.value()).into();
        let rep = (v + h) % m - h;
        format!("{}", rep)
    }
}

impl ReprMod for FE {}

#[derive(Debug, Clone, Eq)]
pub struct Poly<F>
where
    Self: PartialEq,
{
    coeffs: Vec<F>,
}

impl Poly<FE> {
    fn new() -> Self {
        Self { coeffs: vec![] }
    }

    fn get_coeff(&self, i: usize) -> FE {
        if i < self.coeffs.len() {
            self.coeffs[i]
        } else {
            FE::zero()
        }
    }

    fn zero() -> Self {
        Self::new()
    }

    fn zeros(n: usize) -> Self {
        Self {
            coeffs: vec![FE::zero(); n],
        }
    }

    fn one() -> Self {
        Self::from(vec![FE::one()])
    }

    fn trimtrailingzeros(&self) -> Self {
        let mut res = self.clone();
        for c in self.coeffs.iter().rev() {
            if c != &FE::zero() {
                break;
            }
            res.coeffs.pop();
        }
        res
    }
    fn degree(&self) -> isize {
        self.trimtrailingzeros().coeffs.len() as isize - 1
    }
}

impl Rand for Poly<FE> {
    type Multi = Poly<FE>;

    fn rand() -> Self {
        Poly::rand_n(random::<u64>() % 10)
    }

    fn rand_n(n: u64) -> Poly<FE> {
        let leading = FE::rand_except(&vec![FE::zero()]);
        let mut coeffs = FE::rand_n(n);
        coeffs.push(leading);
        Poly::from(coeffs)
    }
}

impl RandExc<FE> for Poly<FE> {
    type Multi = Self;

    fn rand_except(excluded: &Vec<FE>) -> Self {
        Poly::from(FE::rand_except(excluded))
    }

    fn rand_n_except(n: u64, excluded: &Vec<FE>) -> Self::Multi {
        Poly::from(FE::rand_n_except(n, excluded))
    }
}

impl RandExc<Poly<FE>> for Poly<FE> {
    type Multi = Self;

    fn rand_except(excluded: &Vec<Poly<FE>>) -> Self {
        let mut res = Poly::rand();
        while excluded.iter().any(|excl| res == *excl) {
            res = Poly::rand();
        }
        res
    }

    fn rand_n_except(n: u64, excluded: &Vec<Poly<FE>>) -> Self::Multi {
        let mut res = Poly::rand_n(n);
        while excluded.iter().any(|excl| res == *excl) {
            res = Poly::rand_n(n);
        }
        res
    }
}

impl Repr for Poly<FE> {
    fn repr(&self) -> String {
        let mut s = String::new();
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c == &FE::zero() {
                continue;
            }
            let mut rc = c.repr();
            let mut sign = " + ";
            if !first && rc.starts_with('-') {
                sign = " - ";
                rc = rc[1..].to_string();
            }
            match first {
                true => first = false,
                false => s.push_str(sign),
            }
            match i {
                0 => s.push_str(&format!("{:>2}", rc)),
                1 => s.push_str(&format!("{:>2}x", rc)),
                _ => s.push_str(&format!("{:>2}x^{:<2}", rc, i)),
            }
        }
        if first {
            s.push('0');
        }
        s
    }
    fn dbg(&self) -> String {
        let mut res = String::new();
        res.push('[');
        for (i, c) in self.coeffs.iter().enumerate() {
            match i {
                0 => res.push_str(&c.repr()),
                _ => res.push_str(&format!(", {}", c.repr())),
            }
        }
        res.push(']');
        res
    }
}

impl ReprMod for Poly<FE> {}

impl<T> From<Vec<T>> for Poly<FE>
where
    FE: From<T>,
{
    fn from(v: Vec<T>) -> Self {
        let mut coeffs = vec![];
        for i in v {
            coeffs.push(FE::from(i));
        }
        Self { coeffs }
    }
}

impl From<i64> for Poly<FE> {
    fn from(i: i64) -> Self {
        let m = MODULUS as i64;
        let res = (i % m + m) % m;
        Self::from(vec![res as u64])
    }
}

impl From<FE> for Poly<FE> {
    fn from(value: FE) -> Self {
        Self::from(vec![value])
    }
}

impl Display for Poly<FE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr_mod())
    }
}

impl Add for Poly<FE> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut coeffs = vec![];
        let longest = max(self.coeffs.len(), rhs.coeffs.len());
        for i in 0..longest {
            let a = self.get_coeff(i);
            let b = rhs.get_coeff(i);
            coeffs.push(a + b);
        }
        Self::from(coeffs)
    }
}

impl Mul for Poly<FE> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Poly::zeros(self.coeffs.len() + rhs.coeffs.len() + 1);
        for (l_exp, l_coeff) in self.coeffs.iter().enumerate() {
            for (r_exp, r_coeff) in rhs.coeffs.iter().enumerate() {
                res.coeffs[l_exp + r_exp] += l_coeff * r_coeff;
            }
        }
        res.trimtrailingzeros()
    }
}

impl Neg for Poly<FE> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Poly::from(-1) * self
    }
}

impl Sub for Poly<FE> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl DivMod<Poly<FE>> for Poly<FE> {
    fn divmod(&self, other: Poly<FE>) -> (Self, Self)
    where
        Self: Sized,
    {
        let lhs = Box::new(self.trimtrailingzeros());
        let rhs = Box::new(other.trimtrailingzeros());
        if lhs.coeffs.is_empty() {
            return (Poly::new(), Poly::new());
        }
        let mut rem = lhs;
        let (mut lenl, mut lenr) = (rem.coeffs.len(), other.coeffs.len());
        let mut quotient = if lenl >= lenr {
            Poly::zeros(lenl - lenr + 1)
        } else {
            Poly::new()
        };
        let g_msc_inv = &rhs.coeffs.last().unwrap().inv().unwrap();
        while lenl >= lenr {
            let deg_dif = lenl - lenr;
            let tmp = rem.coeffs.last().unwrap() * g_msc_inv;
            quotient.coeffs[deg_dif] += tmp;
            let mut last_non_zero: i64 = deg_dif as i64 - 1;
            for (i, coef) in rhs.coeffs.iter().enumerate() {
                let i = i + deg_dif;
                rem.coeffs[i] = rem.coeffs[i] - tmp * coef;
                if rem.coeffs[i] != FE::zero() {
                    last_non_zero = i as i64;
                }
            }
            let mut rem_coeffs = vec![];
            for rem_coef in rem.coeffs.iter().take((last_non_zero + 1) as usize) {
                rem_coeffs.push(*rem_coef)
            }
            *rem = Poly::from(rem_coeffs);
            (lenl, lenr) = (rem.coeffs.len(), other.coeffs.len());
        }
        (quotient.trimtrailingzeros(), rem.trimtrailingzeros())
    }
}

impl Div for Poly<FE> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        DivMod::div(&self, rhs)
    }
}

impl Rem for Poly<FE> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        DivMod::modulus(&self, rhs)
    }
}

impl PartialEq for Poly<FE> {
    fn eq(&self, other: &Self) -> bool {
        self.trimtrailingzeros().coeffs == other.trimtrailingzeros().coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_repr() {
        type P = Poly<FE>;
        for i in 0..MODULUS {
            let fe = FE::from(i);
            println!("{}: {:?}", i, fe);
            println!("{}", fe.repr());
        }
        let p = P::from(vec![1, 2, 3]);
        println!("{:?}", p);
        println!("{}", p.repr());
        let p = P::from(vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        ]);
        println!("{:?}", p);
        println!("{}", p.repr());
        let p = P::from(vec![
            0,
            MODULUS - 1,
            MODULUS - 2,
            MODULUS - 3,
            MODULUS - 4,
            MODULUS - 5,
            MODULUS - 6,
            MODULUS - 7,
            MODULUS - 8,
            MODULUS - 9,
            MODULUS - 10,
            MODULUS - 11,
            MODULUS - 12,
            MODULUS - 13,
            MODULUS - 14,
            MODULUS - 15,
            MODULUS - 16,
            MODULUS - 17,
        ]);
        println!("{:?}", p);
        println!("{}", p.repr());
    }

    fn test_rand_poly() {
        for i in 0..2048 {
            println!("--- i = {} ---", i);
            let pn = Poly::rand_n(10);
            println!("deg = {}, pn = {}", pn.degree(), pn);
            assert_eq!(pn.degree(), 10);
        }
    }

    fn test_poly_add() {
        println!("--- same length ---");
        let p1 = Poly::from(vec![1, 2, 3]);
        let p2 = Poly::from(vec![4, 5, 6]);
        println!("{}\n+\n{}", p1, p2);
        let res = p1 + p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![5, 7, 9]));

        println!("--- rhs longer ---");
        let p1 = Poly::from(vec![1, 2]);
        let p2 = Poly::from(vec![4, 5, 6]);
        println!("{}\n+\n{}", p1, p2);
        let res = p1 + p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![5, 7, 6]));

        println!("--- rhs shorter ---");
        let p1 = Poly::from(vec![1, 2, 3]);
        let p2 = Poly::from(vec![4, 5]);
        println!("{}\n+\n{}", p1, p2);
        let res = p1 + p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![5, 7, 3]));
    }

    fn test_poly_mul() {
        println!("--- same length ---");
        let p1 = Poly::from(vec![1, 2, 3]);
        let p2 = Poly::from(vec![4, 5, 6]);
        println!("{}\n*\n{}", p1, p2);
        let res = p1 * p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![4, 13, 28, 27, 18]));

        println!("--- rhs longer ---");
        let p1 = Poly::from(vec![1, 2]);
        let p2 = Poly::from(vec![4, 5, 6]);
        println!("{}\n*\n{}", p1, p2);
        let res = p1 * p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![4, 13, 16, 12]));

        println!("--- rhs shorter ---");
        let p1 = Poly::from(vec![1, 2, 3]);
        let p2 = Poly::from(vec![4, 5]);
        println!("{}\n*\n{}", p1, p2);
        let res = p1 * p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![4, 13, 22, 15]));
    }

    fn test_poly_neg() {
        let p = Poly::from(vec![1, 2, 3]);
        println!("-({})", p);
        let np = -p;
        println!("={}", np);
        assert_eq!(np, Poly::from(vec![MODULUS - 1, MODULUS - 2, MODULUS - 3]));
    }

    fn test_poly_sub() {
        let p1 = Poly::from(vec![1, 2, 3]);
        let p2 = Poly::from(vec![4, 5, 6]);
        println!("{}\n-\n{}", p1, p2);
        let res = p1 - p2;
        println!("=\n{}", res);
        assert_eq!(res, Poly::from(vec![MODULUS - 3, MODULUS - 3, MODULUS - 3]));
    }

    fn test_poly_divmod() {
        let p1 = Poly::from(vec![1, 2, 3, 4]);
        let p2 = Poly::from(vec![1, 2, 3]);
        println!("{}\n/\n{}", p1, p2);
        let (q, r) = p1.clone().divmod(p2.clone());
        println!("=\n({}) * ({}) + {}", q, p2, r);
        assert_eq!(q * p2 + r.clone(), p1);
    }

    pub fn test_divmod_random_poly() {
        for i in 0..1_000 {
            println!("--- i = {} ---", i);
            let deg_a = random::<u64>() % 255;
            let deg_b = random::<u64>() % 255;
            println!("deg_a = {deg_a}");
            println!("deg_b = {deg_b}");
            let a = Poly::rand_n_except(deg_a, &vec![Poly::zero()]);
            let b = Poly::rand_n_except(deg_b, &vec![Poly::zero()]);
            println!("a = {a}");
            println!("b = {b}");
            let (q, r) = a.clone().divmod(b.clone());
            let d = r.clone() + q.clone() * b.clone();
            println!(
                "assert!(r.degree() < b.degree()): {} < {}",
                r.clone().degree(),
                b.clone().degree()
            );
            assert!(r.degree() < b.degree());
            println!("assert!(d == a): {} == {}", d, a);
            assert!(d == a);
        }
    }

    #[test]
    fn test_sequence() {
        println!("\n\n=== TEST REPR ===");
        test_repr();
        println!("\n\n=== TEST RAND POLY ===");
        test_rand_poly();
        println!("\n\n=== TEST POLY ADD ===");
        test_poly_add();
        println!("\n\n=== TEST POLY MUL ===");
        test_poly_mul();
        println!("\n\n=== TEST POLY NEG ===");
        test_poly_neg();
        println!("\n\n=== TEST POLY SUB ===");
        test_poly_sub();
        println!("\n\n=== TEST POLY DIVMOD ===");
        test_poly_divmod();
        println!("\n\n=== TEST DIV RANDOM POLY ===");
        test_divmod_random_poly();
        println!("\n\n=== ALL TESTS PASSED ===");
    }
}
