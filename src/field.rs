use ark_ff::fields::{Fp64, MontBackend, MontConfig};
#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
pub struct FqConfig;
pub type FE = Fp64<MontBackend<FqConfig, 1>>;
pub const MOD: u64 = 18446744069414584321;
pub fn gen() -> FE {
    FqConfig::GENERATOR
}
