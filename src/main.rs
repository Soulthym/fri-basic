use std::cmp::max;
use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use lambdaworks_math::field::{
    element::*,
    fields::{
        fft_friendly::stark_252_prime_field::*,
        u64_prime_field::{F17, FE17},
    },
    traits::{IsField, IsPrimeField},
};

//type F = Stark252PrimeField;
//type FE = FieldElement<F>;
type F = F17;
type FE = FE17;
const MODULUS: u64 = 17;

trait Repr {
    fn repr(&self) -> String;
}

trait ReprMod: Repr {
    fn repr_mod(&self) -> String {
        format!("{} % {}", self.repr(), MODULUS)
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

#[derive(Debug, Clone, PartialEq)]
struct Poly<F> {
    coeffs: Vec<F>,
}

impl Poly<FE> {
    fn new() -> Self {
        Self { coeffs: vec![] }
    }

    fn get_coeff(&self, i: usize) -> FE {
        if i < self.coeffs.len() {
            self.coeffs[i].clone()
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
        let mut res = Poly::zeros(self.coeffs.len() + rhs.coeffs.len() - 1);
        for (l_exp, l_coeff) in self.coeffs.iter().enumerate() {
            for (r_exp, r_coeff) in rhs.coeffs.iter().enumerate() {
                res.coeffs[l_exp + r_exp] += l_coeff * r_coeff;
            }
            println!("({}) * ({}) = {}", self, rhs, res);
        }
        res
    }
}

impl Neg for Poly<FE> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Poly::from(-1) * self
    }
}

fn main() {}

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

    #[test]
    fn test_all() {
        println!("\n\n=== TEST REPR ===");
        test_repr();
        println!("\n\n=== TEST POLY ADD ===");
        test_poly_add();
        println!("\n\n=== TEST POLY MUL ===");
        test_poly_mul();
        println!("\n\n=== TEST POLY NEG ===");
        test_poly_neg();
    }
}
