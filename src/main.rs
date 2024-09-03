#![allow(dead_code)]
mod arithmetic;
mod field;
mod traits;
use arithmetic::*;
use field::*;
use traits::*;

type P = Poly<FE>;

fn is_power_of_2(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

fn order(g: &FE) -> u64 {
    let mut order = 1;
    let mut x = *g;
    while x != FE::from(1) {
        x *= g;
        order += 1;
    }
    order
}

fn subgen(size: u64) -> FE {
    assert!(
        is_power_of_2(size) || size == 1,
        "size must be a power of 2"
    );
    let subgen = gen().pow((MOD - 1) / size);
    let ord = order(&subgen);
    assert_eq!(ord, size, "subgen of order {} instead of {}", ord, size);
    subgen
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subgen() {
        let mut size = 1;
        for i in 0..16 {
            size *= 2;
            let subgen = subgen(size);
            println!("subgen(2^{i} = {}): {}", size, subgen);
            let ord = order(&subgen);
            println!("subgen of order {}, expected {}", ord, size);
        }
    }
}
