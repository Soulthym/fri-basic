use ark_ff::fields::{Field, Fp64, MontBackend, MontConfig};
use rand::{rngs::ThreadRng, thread_rng};

trait Rand {
    fn rand(rng: &mut ThreadRng) -> Self;
}

trait RandN {
    fn rand(rng: &mut ThreadRng, num: u64) -> Self;
}

#[derive(MontConfig)]
#[modulus = "17"]
#[generator = "3"]
pub struct SmallFieldConfig;
pub type SmallField = Fp64<MontBackend<SmallFieldConfig, 1>>;

#[derive(MontConfig)]
#[modulus = "3221225473"]
#[generator = "5"]
pub struct StarkFieldConfig;
pub type StarkField = Fp64<MontBackend<StarkFieldConfig, 1>>;

impl<T: Field> Rand for T {
    fn rand(rng: &mut ThreadRng) -> Self {
        T::rand(rng)
    }
}

type Fq = SmallField;

fn main() {
    type Fq = StarkField;
    let p: Poly<Fq> = vec![rand_field::<Fq>(); 8];
    println!("p = {:?}", p);
}

#[cfg(test)]
mod tests {
    #[test]
    fn small_field() {
        use super::*;

        type F = SmallField;

        let a = F::from(9);
        let b = F::from(10);

        println!("a = {:?}", a); // -1 = 16 mod 17
        println!("b = {:?}", b); // -1 = 16 mod 17
        println!("26 = {:?} [expect: 9]", F::from(26)); // 26 =  9 mod 17
        println!("a - b = {:?} [expect: 16]", a - b); // -1 = 16 mod 17
        println!("a + b = {:?} [expect: 2]", a + b); // 19 =  2 mod 17
        println!("a * b = {:?} [expect: 5]", a * b); // 90 =  5 mod 17
        println!("a.square() = {:?} [expect: 13]", a.square()); // 81 = 13 mod 17
        println!("b.double() = {:?} [expect: 3]", b.double()); // 20 =  3 mod 17
        println!("a / b = {:?} [expect: 6]", a / b);
        println!("a ** b = {:?} [expect: 13]", a.pow(b.0)); // pow takes BigInt as input
        assert_eq!(a, F::from(26)); // 26 =  9 mod 17
        assert_eq!(a - b, F::from(16)); // -1 = 16 mod 17
        assert_eq!(a + b, F::from(2)); // 19 =  2 mod 17
        assert_eq!(a * b, F::from(5)); // 90 =  5 mod 17
        assert_eq!(a.square(), F::from(13)); // 81 = 13 mod 17
        assert_eq!(b.double(), F::from(3)); // 20 =  3 mod 17
        assert_eq!(a / b, a * b.inverse().unwrap()); // need to unwrap since `b` could be 0 which is not invertible
        assert_eq!(a.pow(b.0), F::from(13)); // pow takes BigInt as input
    }
}
