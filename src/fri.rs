use crate::field::*;
use crate::poly::*;
use crate::traits::*;
use ark_ff::{BigInteger, FftField, One, PrimeField};
use rand::random;
use rs_merkle::{algorithms::Sha256, Hasher, MerkleProof, MerkleTree};

pub type Merkle = MerkleTree<Sha256>;
pub type MerklePath = MerkleProof<Sha256>;

pub type P = Poly<FE>;

pub enum Message {
    Commit(String),
    Random(FE),
    RandomInt(u64),
    Answer(FE),
    Proof(MerklePath),
    NewLayer(u64),
    Begin(String),
    Finish,
}

pub trait TranscriptProtocol {
    fn commit(&mut self, value: String);
    fn random(&mut self) -> FE;
    fn random_int(&mut self, min: u64, max: u64) -> u64;
    fn answer(&mut self, value: &FE);
    fn prove(&mut self, proof: MerklePath);
    fn new_layer(&mut self, length: u64);
    fn begin(&mut self, value: String);
    fn finish(&mut self);
    fn serialize(&self) -> String;
    fn show(&self) {
        let serialized = self.serialize();
        println!(
            "{}\nUncompressed length: {}\n",
            serialized,
            serialized.len()
        );
    }
}

pub type Transcript = Vec<Message>;

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

    fn new_layer(&mut self, length: u64) {
        self.push(Message::NewLayer(length));
    }

    fn begin(&mut self, value: String) {
        self.push(Message::Begin(value));
    }

    fn finish(&mut self) {
        self.push(Message::Finish);
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
                Message::NewLayer(value) => {
                    serialized.push_str(&format!("\nNewLayer len:{}\n", value))
                }
                Message::Begin(value) => {
                    serialized.push_str(&format!("\n===== Begin: {} =====\n", value))
                }
                Message::Finish => serialized.push_str("\n===== Finish =====\n"),
            }
        }
        serialized
    }
}

fn generate_eval_domain(log2size: u64) -> Vec<FE> {
    let g = FE::generate_coset_log2size(log2size);
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

fn commit_layer(transcript: &mut Transcript, poly_evals: &Vec<FE>, oracle: &mut Merkle) {
    let leaves = poly_evals
        .iter()
        .map(|&elem| Sha256::hash(elem.to_string().as_bytes()))
        .collect::<Vec<_>>();
    for &elem in &leaves {
        oracle.insert(elem);
    }
    oracle.commit();
    let root_hex = oracle.root_hex().unwrap();
    transcript.commit(root_hex);
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

pub fn fri_commit(
    transcript: &mut Transcript,
    poly: &P,
    domain: &Vec<FE>,
    layer: &Vec<FE>,
    oracle: &mut Merkle,
) -> (Vec<P>, Vec<Vec<FE>>, Vec<Vec<FE>>, Vec<Merkle>) {
    transcript.begin("fri_commit".to_string());
    let mut fri_polys: Vec<P> = vec![poly.clone()];
    let mut fri_domains: Vec<Vec<FE>> = vec![domain.clone()];
    let mut fri_layers: Vec<Vec<FE>> = vec![layer.clone()];
    let mut fri_oracles: Vec<Merkle> = vec![oracle.clone()];
    while fri_polys.last().unwrap().degree() > 0 {
        transcript.new_layer(fri_layers.last().unwrap().len() as u64);
        let beta = transcript.random();
        let (next_poly, next_domain, next_layer) =
            next_fri_layer(fri_polys.last().unwrap(), fri_domains.last().unwrap(), beta);
        fri_polys.push(next_poly);
        fri_domains.push(next_domain);
        fri_layers.push(next_layer.clone());
        let mut next_oracle = Merkle::new();
        commit_layer(transcript, &next_layer, &mut next_oracle);
        fri_oracles.push(next_oracle);
    }
    let last_value = fri_layers.last().unwrap().first().unwrap();
    transcript.answer(last_value);
    (fri_polys, fri_domains, fri_layers, fri_oracles)
}

fn fri_decommit_on_layers(
    transcript: &mut Transcript,
    idx: u64,
    fri_layers: &Vec<Vec<FE>>,
    fri_oracles: &Vec<Merkle>,
) {
    transcript.begin(format!("fri_decommit layers on {}", idx).to_string());
    let mut idx = idx;
    for (layer, oracle) in fri_layers
        .iter()
        .zip(fri_oracles.iter())
        .take(fri_layers.len() - 1)
    {
        let length = layer.len() as u64;
        transcript.new_layer(length);
        idx %= length;
        let sibling_idx = (idx + length / 2) % length;
        transcript.answer(&layer[idx as usize]);
        transcript.prove(oracle.proof(&[idx as usize]));
        transcript.answer(&layer[sibling_idx as usize]);
        transcript.prove(oracle.proof(&[sibling_idx as usize]));
    }
    let const_fri_layer = fri_layers.last().unwrap();
    let length = const_fri_layer.len() as u64;
    transcript.new_layer(length);
    transcript.answer(&const_fri_layer[0]);
}

fn fri_decommit_on_query(
    transcript: &mut Transcript,
    idx: u64,
    log2blowup: &u64,
    root_layer: &Vec<FE>,
    root_oracle: &Merkle,
    fri_layers: &Vec<Vec<FE>>,
    fri_oracles: &Vec<Merkle>,
) {
    transcript.begin(format!("fri_decommit query on {}", idx).to_string());
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
        transcript.prove(root_oracle.proof(&[idx as usize]));
    }
    fri_decommit_on_layers(transcript, idx, fri_layers, fri_oracles);
    transcript.finish();
}

pub fn fri_decommit<'a>(
    transcript: &mut Transcript,
    log2blowup: &'a u64,
    root_layer: &'a Vec<FE>,
    root_oracle: &'a Merkle,
    fri_layers: &'a Vec<Vec<FE>>,
    fri_oracles: &'a Vec<Merkle>,
) -> &'a Vec<Merkle> {
    let blowup = 1 << log2blowup;
    for i in 0..1 {
        transcript.begin(format!("fri_decommit[{i}]"));
        let idx = transcript.random_int(0, root_layer.len() as u64 - blowup * 2);
        fri_decommit_on_query(
            transcript,
            idx,
            log2blowup,
            root_layer,
            root_oracle,
            fri_layers,
            fri_oracles,
        );
    }
    fri_oracles
}

type TranscriptData<'a> = (
    String,
    Vec<String>,
    Vec<FE>,
    FE,
    u64,
    Vec<FE>,
    Vec<&'a MerklePath>,
    Vec<(Vec<FE>, Vec<&'a MerklePath>)>,
);

pub fn fri_parse_transcript<'a>(
    transcript: &'a Transcript,
    log2blowup: &'a u64,
) -> TranscriptData<'a> {
    let blowup = 1 << log2blowup;
    let mut original_commit: String = String::new();

    println!("\nParsing original commit");
    let mut iter_transcript = transcript.iter();
    for message in iter_transcript.by_ref() {
        match message {
            Message::Commit(value) => {
                original_commit = value.to_string();
            }
            Message::Begin(_) => break,
            _ => {}
        }
    }

    println!("\nParsing commit phase");
    let mut commits: Vec<String> = vec![];
    let mut betas: Vec<FE> = vec![];
    let mut final_answer: FE = FE::from(0);
    for message in iter_transcript.by_ref() {
        match message {
            Message::Commit(value) => commits.push(value.to_string()),
            Message::Random(value) => betas.push(*value),
            Message::Answer(value) => {
                final_answer = *value;
            }
            Message::Begin(_) => break,
            _ => {}
        }
    }

    println!("\nParsing decommit query phase");
    let mut random_int: u64 = u64::MAX;
    for message in iter_transcript.by_ref() {
        match message {
            Message::RandomInt(value) => random_int = *value,
            Message::Begin(_) => break,
            _ => {}
        }
    }
    println!("Random int: {:?}", random_int);
    let mut root_answers: Vec<FE> = vec![];
    let mut root_proofs: Vec<&MerklePath> = vec![];
    let mut layer_size: u64 = 0;
    for message in iter_transcript.by_ref() {
        match message {
            Message::Answer(value) => root_answers.push(*value),
            Message::Proof(proof) => root_proofs.push(proof),
            Message::NewLayer(value) => {
                layer_size = *value;
                break;
            }
            _ => {}
        }
    }
    assert!(random_int != u64::MAX, "Random int not found");
    assert_eq!(root_answers.len(), 3);
    assert_eq!(root_proofs.len(), 3);
    assert!(layer_size > 0);

    println!("\nParsing decommit layers phase");
    let mut layer_steps: Vec<(Vec<FE>, Vec<&'a MerklePath>)> = vec![];
    while layer_size >= blowup {
        let mut answers: Vec<FE> = vec![];
        let mut proofs: Vec<&'a MerklePath> = vec![];
        if iter_transcript.len() == 0 {
            break;
        }
        for message in iter_transcript.by_ref() {
            match message {
                Message::NewLayer(layer_len) => {
                    layer_size = *layer_len;
                    if !answers.is_empty() || !proofs.is_empty() {
                        layer_steps.push((answers, proofs));
                    }
                    break;
                }
                Message::Answer(value) => {
                    answers.push(*value);
                }
                Message::Proof(proof) => {
                    proofs.push(proof);
                }
                Message::Finish => {
                    if !answers.is_empty() || !proofs.is_empty() {
                        layer_steps.push((answers, proofs));
                    }
                    break;
                }
                _ => {}
            }
        }
    }

    for (i, layer) in layer_steps.iter().enumerate() {
        let answers = &layer.0;
        let proofs = &layer.1;
        if i == layer_steps.len() - 1 {
            assert_eq!(answers.len(), 1);
            assert_eq!(proofs.len(), 0);
        } else {
            assert_eq!(answers.len(), 2);
            assert_eq!(proofs.len(), 2);
        }
    }
    println!("original_commit = {}", original_commit);
    println!("commits = {:?}", commits);
    println!("betas = {:?}", betas);
    println!("final_answer = {}", final_answer);
    println!("random_int = {}", random_int);
    println!("root_answers = {:?}", root_answers);
    println!("root_proofs : {}", root_proofs.len());
    println!("layer_steps : {}", layer_steps.len());
    (
        original_commit,
        commits,
        betas,
        final_answer,
        random_int,
        root_answers,
        root_proofs,
        layer_steps,
    )
}

fn fri_verify(
    transcript: &Transcript,
    log2blowup: &u64,
    log2size_extended: &u64,
    oracles: &Vec<Merkle>,
) -> bool {
    let (
        _original_commit,
        _commits,
        betas,
        _final_answer,
        random_int,
        _root_answers,
        _root_proofs,
        layer_steps,
    ) = fri_parse_transcript(transcript, log2blowup);
    println!("\nVerifying...");
    let mut domain = generate_eval_domain(*log2size_extended);
    let mut idx = random_int;

    for (i, layer) in layer_steps.iter().enumerate().take(layer_steps.len() - 1) {
        let (answers, proofs) = layer;
        assert_eq!(answers.len(), 2, "Expected 2 answers");
        assert_eq!(proofs.len(), 2, "Expected 2 proofs");
        // check that foldings are correct
        // namely, that the answer is the correct folding of the previous layer
        // with the beta value evaluated at recorded element
        // let p be the previous layer, and f be the current layer
        // f(beta) = p0(beta²) + beta * p1(beta²)
        // f(-beta) = p0(beta²) - beta * p1(beta²)
        // f0 = f(beta) is the answer at the current layer
        // f1 = f(-beta) is the sibling answer at the current layer
        // p0 = p(beta²) is the answer at the previous layer
        // p1 = p(beta²) is the sibling answer at the previous layer
        // so we have:
        // f0 = p0 + beta * p1
        // f1 = p0 - beta * p1
        // we solve for p0 and p1:
        // p0 = (f0 + f1) / 2
        // p1 = (f0 - f1) / (2 * beta)
        // we check that p0 and p1 are the correct answers at the previous layer
        let beta = betas[i];
        let oracle = oracles.get(i).expect("Failed to get oracle");

        let length = domain.len() as u64;
        idx %= length;
        let sibling_idx = (idx + length / 2) % length;
        let f0 = answers.get(0).expect("Failed to get answer");
        let f1 = answers.get(1).expect("Failed to get answer");
        let p0 = (f0 + f1) / FE::from(2);
        let p1 = (f0 - f1) / (FE::from(2) * beta);
        let res = p0 + p1 * beta;
        if res != *f0 {
            println!(
                "Folding consistency check failed at layer {} for index {}",
                i, 0
            );
            println!("f0 = {f0}");
            println!("f1 = {f1}");
            println!("p0 = {p0}");
            println!("p1 = {p1}");
            println!("p0 + p1 * beta = {res} != f0 = {f0}");
            return false;
        }

        // check that the merkle proofs are correct
        if !verify_merkle_proof(
            proofs.get(0).expect("Failed to get proof"),
            idx,
            &res,
            oracle,
            length as usize,
        ) {
            println!(
                "Merkle proof verification failed for f0 at layer {} for answer {}",
                i, 0
            );
            println!(
                "proof = {:?}",
                proofs
                    .get(0)
                    .expect("Failed to get proof")
                    .proof_hashes_hex()
            );
            println!("f0 = {f0}");
            println!("idx = {idx}");
            return false;
        }
        if !verify_merkle_proof(
            proofs.get(1).expect("Failed to get proof"),
            sibling_idx,
            f1,
            oracle,
            length as usize,
        ) {
            println!(
                "Merkle proof verification failed for f1 at layer {} for answer {}",
                i, 0
            );
            println!(
                "proof = {:?}",
                proofs
                    .get(0)
                    .expect("Failed to get proof")
                    .proof_hashes_hex()
            );
            println!("f0 = {f0}");
            println!("idx = {idx}");
            return false;
        }

        domain = next_fri_domain(&domain);
    }
    true
}

fn verify_merkle_proof(
    proof: &MerklePath,
    idx: u64,
    leaf: &FE,
    oracle: &Merkle,
    len_leaves: usize,
) -> bool {
    let indices_to_prove: Vec<usize> = vec![idx as usize];
    let leaves_to_prove = &[Sha256::hash(leaf.to_string().as_bytes())];
    let root = oracle.root().expect("Failed to get root");
    proof.verify(root, &indices_to_prove, leaves_to_prove, len_leaves)
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

    #[test]
    fn test_merkle() {
        let elems = [1, 2, 3, 4, 5, 6, 7, 8]
            .iter()
            .map(|&x| FE::from(x))
            .collect::<Vec<_>>();
        println!("elems = {:?}", elems);
        let leaves = elems
            .iter()
            .map(|&elem| Sha256::hash(elem.to_string().as_bytes()))
            .collect::<Vec<_>>();
        println!("Elems: {:?}", elems);
        let mut merkle = Merkle::new();
        for &elem in &elems {
            merkle.insert(Sha256::hash(elem.to_string().as_bytes()));
        }
        merkle.commit();
        let indices_to_prove = vec![1];
        let leaves_to_prove = &[*leaves.get(1).unwrap()];
        let root = merkle.root().expect("Failed to get root");
        let root_hex = merkle.root_hex().expect("Failed to get root hex");
        println!("Root: {}", root_hex);
        let proof = merkle.proof(&indices_to_prove);
        println!("Proof: {:?}", proof.proof_hashes_hex());
        println!(
            "Verify merkle proof: {}",
            proof.verify(root, &indices_to_prove, leaves_to_prove, leaves.len())
        );
        println!(
            "Verify merkle proof: {}",
            verify_merkle_proof(&proof, 1, &FE::from(2), &merkle, leaves.len())
        );
    }

    #[test]
    fn test_fri() {
        let log2size: u64 = 10;
        let len: u64 = 1 << log2size;
        let log2blowup: u64 = 3;
        let log2size_extended: u64 = log2size + log2blowup;
        let poly = P::rand_n(len);
        let mut oracle = Merkle::new();
        let eval_domain = generate_eval_domain(log2size_extended);
        let root_layer = eval_on_domain(&poly, &eval_domain);
        let mut transcript: Transcript = vec![];
        commit_layer(&mut transcript, &root_layer, &mut oracle);
        let fri_domain = eval_domain.clone();
        let (_fri_polys, _fri_domains, fri_layers, fri_oracles) = fri_commit(
            &mut transcript,
            &poly,
            &fri_domain,
            &root_layer,
            &mut oracle,
        );
        let oracles = fri_decommit(
            &mut transcript,
            &log2blowup,
            &root_layer,
            &oracle,
            &fri_layers,
            &fri_oracles,
        );

        transcript.show();
        assert!(fri_verify(
            &transcript,
            &log2blowup,
            &log2size_extended,
            oracles
        ));
    }
}
