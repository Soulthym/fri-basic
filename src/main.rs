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

type P = Poly<FE>;

enum Message {
    Commit(String),
    Random(FE),
    RandomInt(u64),
    Answer(FE),
    Proof(MerklePath),
}

trait TranscriptProtocol {
    fn commit(&mut self, value: String);
    fn random(&mut self) -> FE;
    fn random_int(&mut self, min: u64, max: u64) -> u64;
    fn answer(&mut self, value: &FE);
    fn prove(&mut self, proof: MerklePath);
    fn serialize(&self) -> String;
    fn show(&self) {
        let serialized = self.serialize();
        println!("{}\nUncompressed length: {}", serialized, serialized.len());
    }
}

type Transcript = Vec<Message>;

impl TranscriptProtocol for Transcript {
    fn commit(&mut self, value: String) {
        self.push(Message::Commit(value));
    }

    fn random(&mut self) -> FE {
        let value = FE::rand();
        self.push(Message::Random(value));
        value
    }

    fn random_int(&mut self, min: u64, max: u64) -> u64 {
        assert!(min < max, "min={} must be less than max={}", min, max);
        let delta = max - min + 1;
        let value = random::<u64>() % delta + min;
        self.push(Message::RandomInt(value));
        value
    }

    fn answer(&mut self, value: &FE) {
        self.push(Message::Answer(*value));
    }

    fn prove(&mut self, proof: MerklePath) {
        self.push(Message::Proof(proof));
    }

    fn serialize(&self) -> String {
        let mut serialized = String::new();
        serialized.push_str("Transcript:\n");
        for message in self {
            match message {
                Message::Commit(value) => serialized.push_str(&format!("Commit: {}\n", value)),
                Message::Random(value) => serialized.push_str(&format!("Random: {}\n", value)),
                Message::RandomInt(value) => {
                    serialized.push_str(&format!("RandomInt: {}\n", value))
                }
                Message::Answer(value) => serialized.push_str(&format!("Answer: {}\n", value)),
                Message::Proof(proof) => {
                    serialized.push_str(&format!("Proof: {:?}\n", proof.proof_hashes_hex()))
                }
            }
        }
        serialized
    }
}

fn generate_eval_domain(log2size: u64) -> Vec<FE> {
    let g = FE::subgen(log2size);
    let len = 1 << log2size;
    (0..len).map(|i| g.pow(i)).collect()
}

fn eval_on_domain(poly: &P, domain: &Vec<FE>) -> Vec<FE> {
    let mut poly_evals: Vec<FE> = vec![];
    for &x in domain {
        let y = poly.eval(x);
        poly_evals.push(y);
    }
    poly_evals
}

fn commit_layer(transcript: &mut Transcript, poly_evals: &Vec<FE>, merkle: &mut Merkle) {
    let leaves = poly_evals
        .iter()
        .map(|&elem| Sha256::hash(elem.to_string().as_bytes()))
        .collect::<Vec<_>>();
    for &elem in &leaves {
        merkle.insert(elem);
    }
    merkle.commit();
    let root_hex = merkle.root_hex().unwrap();
    transcript.commit(root_hex);
    //poly_evals
    //    .iter()
    //    .for_each(|&elem| transcript.commit(elem.to_string()));
}

fn next_fri_domain(domain: &Vec<FE>) -> Vec<FE> {
    domain
        .iter()
        .take(domain.len() / 2)
        .map(|&x| x.pow(2))
        .collect()
}

fn fri_fold_poly(poly: &P, beta: FE) -> P {
    let mut even_coeffs: Vec<FE> = vec![];
    let mut odd_coeffs: Vec<FE> = vec![];
    for (i, &coeff) in poly.coeffs.iter().enumerate() {
        if i % 2 == 0 {
            even_coeffs.push(coeff);
        } else {
            odd_coeffs.push(coeff);
        }
    }
    let even = P::from(even_coeffs);
    let odd = P::from(odd_coeffs);
    even + odd * Poly::from(beta)
}

fn next_fri_layer(poly: &P, domain: &Vec<FE>, beta: FE) -> (P, Vec<FE>, Vec<FE>) {
    let next_poly = fri_fold_poly(poly, beta);
    let next_domain = next_fri_domain(domain);
    let next_layer = eval_on_domain(&next_poly, &next_domain);
    (next_poly, next_domain, next_layer)
}

fn fri_commit(
    transcript: &mut Transcript,
    poly: &P,
    domain: &Vec<FE>,
    layer: &Vec<FE>,
    merkle: &mut Merkle,
) -> (Vec<P>, Vec<Vec<FE>>, Vec<Vec<FE>>, Vec<Merkle>) {
    let mut fri_polys: Vec<P> = vec![poly.clone()];
    let mut fri_domains: Vec<Vec<FE>> = vec![domain.clone()];
    let mut fri_layers: Vec<Vec<FE>> = vec![layer.clone()];
    let mut fri_merkles: Vec<Merkle> = vec![merkle.clone()];
    while fri_polys.last().unwrap().degree() > 0 {
        let beta = transcript.random();
        let (next_poly, next_domain, next_layer) =
            next_fri_layer(fri_polys.last().unwrap(), fri_domains.last().unwrap(), beta);
        fri_polys.push(next_poly);
        fri_domains.push(next_domain);
        fri_layers.push(next_layer.clone());
        let mut next_merkle = Merkle::new();
        commit_layer(transcript, &next_layer, &mut next_merkle);
        fri_merkles.push(next_merkle);
    }
    let last_value = fri_layers.last().unwrap().first().unwrap();
    transcript.answer(last_value);
    (fri_polys, fri_domains, fri_layers, fri_merkles)
}

fn fri_decommit_on_layers(
    transcript: &mut Transcript,
    idx: u64,
    fri_layers: &Vec<Vec<FE>>,
    fri_merkles: &Vec<Merkle>,
) {
    let mut idx = idx;
    for (layer, merkle) in fri_layers
        .iter()
        .zip(fri_merkles.iter())
        .take(fri_layers.len() - 1)
    {
        let length = layer.len() as u64;
        idx %= length;
        let sibling_idx = (idx + length / 2) % length;
        transcript.answer(&layer[idx as usize]);
        transcript.prove(merkle.proof(&[idx as usize]));
        transcript.answer(&layer[sibling_idx as usize]);
        transcript.prove(merkle.proof(&[sibling_idx as usize]));
    }
}

fn fri_decommit_on_query(
    transcript: &mut Transcript,
    idx: u64,
    log2blowup: &u64,
    root_layer: &Vec<FE>,
    root_merkle: &Merkle,
    fri_layers: &Vec<Vec<FE>>,
    fri_merkles: &Vec<Merkle>,
) {
    let blowup = 1 << log2blowup;
    let size_limit = root_layer.len() as u64 - blowup * 2;
    assert!(
        idx < size_limit,
        "idx needs to be less than {}",
        root_layer.len() as u64 - log2blowup
    );
    for i in 0..3 {
        let idx = idx + i * blowup;
        transcript.answer(&root_layer[idx as usize]);
        transcript.prove(root_merkle.proof(&[idx as usize]));
    }
    fri_decommit_on_layers(transcript, idx, fri_layers, fri_merkles);
}

fn fri_decommit(
    transcript: &mut Transcript,
    log2blowup: &u64,
    root_layer: &Vec<FE>,
    root_merkle: &Merkle,
    fri_layers: &Vec<Vec<FE>>,
    fri_merkles: &Vec<Merkle>,
) {
    let blowup = 1 << log2blowup;
    for _ in 0..3 {
        let idx = transcript.random_int(0, root_layer.len() as u64 - blowup * 2);
        fri_decommit_on_query(
            transcript,
            idx,
            log2blowup,
            root_layer,
            root_merkle,
            fri_layers,
            fri_merkles,
        );
    }
}

fn main() {
    let log2size: u64 = 10;
    let len: u64 = 1 << log2size;
    let log2blowup: u64 = 3;
    let log2size_extended: u64 = log2size + log2blowup;
    let poly = P::rand_n(len);
    let mut merkle = Merkle::new();
    println!(
        "Merkle root: {}",
        merkle.root_hex().unwrap_or_else(|| "None".to_string())
    );
    let eval_domain = generate_eval_domain(log2size_extended);
    let root_layer = eval_on_domain(&poly, &eval_domain);
    let mut transcript: Transcript = vec![];
    commit_layer(&mut transcript, &root_layer, &mut merkle);
    let fri_domain = eval_domain;
    println!("Merkle root: {}", merkle.root_hex().unwrap());
    let (fri_polys, fri_domains, fri_layers, fri_merkles) = fri_commit(
        &mut transcript,
        &poly,
        &fri_domain,
        &root_layer,
        &mut merkle,
    );
    fri_decommit(
        &mut transcript,
        &log2blowup,
        &root_layer,
        &merkle,
        &fri_layers,
        &fri_merkles,
    );
    transcript.show();
}

#[cfg(test)]
mod fri_tests {
    use super::*;

    #[test]
    fn test_poly_folding() {
        let poly = P::from(vec![2, 3, 4, 5, 6, 7, 8, 9]);
        println!("poly = {poly}");
        let beta = FE::from(2);
        let folded = fri_fold_poly(&poly, beta);
        println!("folded = {folded}");
        let expected = P::from(vec![2 + 3 * 2, 4 + 5 * 2, 6 + 7 * 2, 8 + 9 * 2]);
        println!("expected = {expected}");
        assert_eq!(folded, expected);
    }
}
