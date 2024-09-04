#![allow(dead_code, unused_imports)]
mod field;
mod fri;
mod poly;
mod traits;
use ark_ff::BigInt;
use field::*;
use fri::*;
use poly::*;
use rand::random;
use rs_merkle::{algorithms::Sha256, Hasher, MerkleTree};
use traits::*;

fn main() {
    println!("fri prover");
}
