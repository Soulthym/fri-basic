# fri-basic
This is a simple implementation of the Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI) algorithm in Rust.  
The original algorithm is described in the paper [Fast Reed-Solomon Interactive Oracle Proofs of Proximity](https://eprint.iacr.org/2017/636.pdf) by Eli Ben-Sasson, Iddo Bentov, Ynon Horesh, and Michael Riabzev  

## Dependencies
Make sure you have rustup and cargo installed on your machine. If not, you can install it by following the instructions [here](https://www.rust-lang.org/tools/install).  

```bash
rustup update
rustup install 1.74.1
```

Clone the repository:  
```bash
git clone https://github.com/Soulthym/fri-basic.git
```

## Running the code
Everything is happening in tests. To run the tests, simply run the following command in the root directory of the project:  
```bash
cargo test
```

if you wish to simply run the FRI tests and show the ouput, you can run the following command:  
```bash
cargo test fri -- --nocapture
```

## Implementation details
I am using the Goldilocks Field 2^64-2^32+1 for the implementation. The field is defined in the `field.rs` file.  
This field's 'magic' values were reverse-engineered from the [Plonky2 codebase](https://github.com/0xPolygonZero/plonky2/blob/main/field/src/goldilocks_field.rs), mainly the multiplicative coset generator.  
I implemented polynomial algebra in the `polynomial.rs` file.  
The FRI prover and verifiers are implemented in the `fri.rs` file.  
The "Fiat Shamir" transform is merely simulated by using RNG to generate the challenges. The workflow I'm using was inspired from it though as it makes testing easier to have it non-interactive.  
There is also no interpolation of any kind, as the focus was on the FRI algorithm itself. I only verify layers for consistency and their proofs.

### Notes and acknowledgements
The code is for experimental/demo purposes only and should not be used in production.  

This was made following the [stark101 tutorial](https://starkware.co/stark-101/), which I highly recommend if you want to understand how STARKs work.  

A few honorable mentions:
- [Anatomy of a STARK, Part 3: FRI](https://aszepieniec.github.io/stark-anatomy/fri.html)
- [Risc Zero's study club - Stark by hand](https://dev.risczero.com/proof-system/stark-by-hand) and their [YouTube channel](https://www.youtube.com/watch?v=j35yz22OVGE&list=PLcPzhUaCxlCjdhONxEYZ1dgKjZh3ZvPtl&index=2&t=0s)
