use std::fmt::Display;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

const MODULUS: i128 = 3 * (1 << 30) + 1;
const GENERATOR: i128 = 5;

trait Inv {
    type Output;
    fn inverse(self) -> Self::Output;
}

trait Pow {
    type Output;
    fn pow(self, n: Self) -> Self::Output;
}

trait Field: Add + Sub + Mul + Pow + Inv + Div + Neg + Rem + Clone + Copy {
    fn new(value: i128) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
}

#[derive(Debug, Clone, Copy)]
struct PrimeField(i128);

impl Display for PrimeField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Field for PrimeField {
    fn new(value: i128) -> Self {
        PrimeField(value % MODULUS)
    }
    fn zero() -> Self {
        PrimeField::new(0)
    }
    fn one() -> Self {
        PrimeField::new(1)
    }
}

impl Add<PrimeField> for PrimeField {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        PrimeField::new(self.0 + other.0)
    }
}

impl Neg for PrimeField {
    type Output = Self;
    fn neg(self) -> Self {
        PrimeField::zero() - self
    }
}

impl Sub<PrimeField> for PrimeField {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        PrimeField::new(self.0 - other.0)
    }
}

impl Mul<PrimeField> for PrimeField {
    type Output = Self;
    fn mul(self, other: PrimeField) -> Self::Output {
        PrimeField::new(self.0 * other.0)
    }
}

impl Pow for PrimeField {
    type Output = Self;
    fn pow(self, mut n: Self) -> Self::Output {
        let mut cur_pow = self;
        let mut res = PrimeField::new(1);
        while n > PrimeField::new(0) {
            if n % PrimeField::new(2) != PrimeField::new(0) {
                res = res * cur_pow;
            }
            n = n / PrimeField::new(2);
            cur_pow = res * cur_pow;
        }
        res
    }
}

impl Inv for PrimeField {
    type Output = Self;
    fn inverse(self) -> Self::Output {
        let (mut t, mut new_t) = (0, 1);
        let (mut r, mut new_r) = (MODULUS, self.0);
        let mut quotient;
        while new_r != 0 {
            quotient = r / new_r;
            (t, new_t) = (new_t, (t - (quotient * new_t)));
            (r, new_r) = (new_r, r - quotient * new_r);
        }
        assert_eq!(r, 1);
        PrimeField::new(t)
    }
}

impl Div<PrimeField> for PrimeField {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Rem<PrimeField> for PrimeField {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        PrimeField::new(self.0 % rhs.0)
    }
}

impl PartialOrd for PrimeField {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrimeField {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialEq for PrimeField {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PrimeField {}

#[cfg(test)]
mod tests {
    use super::*;
    type F = PrimeField;

    #[test]
    fn test_small_prime_field() {
        println!(
            "PrimeField:\n  modulus = {}\n  generator = {}",
            MODULUS, GENERATOR
        );
        let val = F::new(3);
        println!("{}", val);
        //assert_eq!(val, F::new(2));

        let a = F::new(9);
        let b = F::new(-7);

        println!("a = {}", a);
        println!("b = {}", b);
        println!("a + b = {} [expect: {}]", a + b, 2);
        assert_eq!(a + b, F::new(2));
        println!("a - b = {} [expect: {}]", a - b, 16);
        assert_eq!(a - b, F::new(16));
        println!("a * b = {} [expect: {}]", a * b, -63);
        assert_eq!(a * b, F::new(-63));
        println!("a / b = {} [expect: {}]", a / b, -1380525204);
        assert_eq!(a / b, F::new(-1380525204));
        println!("b^a = {} [expect: {}]", b.pow(a), 1);
        assert_eq!(a.pow(b), F::new(1));
    }
}
