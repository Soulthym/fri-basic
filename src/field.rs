use crate::traits::{Pow, Rand, RandExc, Repr, ReprMod, SubGroup};
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
    let base = FE::from(BigInt!("7277203076849721926"));
    base
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
        ark_ff::Field::pow::<&BigInt<1>>(self, &FE::from(n).0)
    }
    fn pow_2_pow(&self, powlog: u64) -> Self::Output {
        let mut res = *self;
        for _ in 0..powlog {
            res *= res
        }
        res
    }
}

impl SubGroup for FE {
    type Output = FE;

    fn order(&self) -> u64 {
        let mut order = 0;
        let mut x = FE::from(1);
        let mut first = true;
        while x != FE::from(1) || first {
            first = false;
            x *= self;
            order += 1;
        }
        order
    }

    fn subgen(log2size: u64) -> FE {
        // based on
        // https://github.com/0xPolygonZero/plonky2/blob/bf95c10cbf44a7759586e22f3b76709bcca3c2ba/field/src/types.rs#L268
        assert!(
            log2size < 32,
            "log2size must be less than the 2-adicity of m: 32"
        );
        gen().pow_2_pow(32 - log2size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgroup() {
        for exp in 0..16 {
            let size = 2_u64.pow(exp as u32);
            let subgen = FE::subgen(exp);
            println!("subgen(2^{exp} = {}): {}", size, subgen);
            let ord = subgen.order();
            println!("subgen of order {}, expected {}", ord, size);
            assert_eq!(ord, size);
        }
    }
}
