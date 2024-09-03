use crate::traits::{Pow, Rand, RandExc, Repr, ReprMod};
use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use ark_ff::{BigInt, PrimeField};
use rand::random;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FqConfig;
pub type FE = Fp64<MontBackend<FqConfig, 1>>;
pub const MOD: u64 = 18446744069414584321;
pub fn gen() -> FE {
    FqConfig::GENERATOR
}

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
        let m = MOD as i128;
        let h = m / 2;
        let v = self.into_bigint().0[0] as i128;
        let rep = (v + h) % m - h;
        format!("{}", rep)
    }
}

impl ReprMod for FE {}

impl Pow<u64> for FE {
    type Output = Self;
    fn pow(&self, n: u64) -> Self::Output {
        ark_ff::Field::pow::<&BigInt<1>>(self, &BigInt::from(n))
    }
}
