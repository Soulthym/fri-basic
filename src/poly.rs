use super::ff::*;
use ark_ff::{BigInt, Field};
use rand::rngs::ThreadRng;
use std::fmt::Display;

pub trait Eval {
    fn eval(self, x: Fq) -> Fq;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<F> {
    coefs: Vec<F>,
}

impl Eval for Poly<Fq> {
    fn eval(self, x: Fq) -> Fq {
        let mut res = Fq::from(0);
        for (expo, coef) in self.coefs.iter().enumerate() {
            match expo {
                0 => res += coef,
                1 => res += x * coef,
                _ => res += x.pow(BigInt::<1>([expo as u64])) * coef,
            }
        }
        res
    }
}

impl RandN for Poly<Fq> {
    fn rand(rng: &mut ThreadRng, len: u64) -> Poly<Fq> {
        let mut coefs = vec![];
        for _ in 0..len {
            coefs.push(Fq::rand(rng));
        }
        Poly { coefs }
    }
}

impl Display for Poly<Fq> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let zero = Fq::from(0);
        for (expo, coef) in self.coefs.iter().enumerate() {
            if *coef == zero {
                continue;
            };
            match expo {
                0 => write!(f, "{:?}", coef.0 .0[0])?,
                1 => write!(f, " + {:?}x", coef.0 .0[0])?,
                _ => write!(f, " + {:?}x^{:?}", coef.0 .0[0], expo)?,
            }
        }
        Ok(())
    }
}
