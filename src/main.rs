use std::fmt::Display;

use ff::derive::bitvec::store::BitStore;

#[macro_use]
extern crate ff;

#[derive(PrimeField)]
#[PrimeFieldModulus = "65537"]
#[PrimeFieldGenerator = "3"]
#[PrimeFieldReprEndianness = "little"]
struct Field([u64; 1]);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Polynomial(Vec<Field>);
impl Display for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (expo, coef_f) in self.0.iter().enumerate() {
            let coef = coef_f.0[0];
            if expo == 0 {
                write!(f, "{:?}x", coef)?;
            } else {
                write!(f, " + {:?}x^{:?}", coef, expo)?;
            }
        }
        Ok(())
    }
}

impl From<String> for Polynomial {
    fn from(value: String) -> Self {
        let mut poly = vec![];
        for byte in value.bytes().into_iter() {
            let value: u64 = byte.into();
            poly.push(Field([value; 1]));
        }
        Polynomial(poly)
    }
}

fn main() {
    let message = String::from("hello world");
    let len = message.len();
    let poly = Polynomial::from(message.clone());
    println!("poly = {}", poly);
    println!("message = {:?}", message);
    println!("len = {:?}", len);
}
