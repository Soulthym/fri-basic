use rs_merkle::{algorithms::Sha256, MerkleProof, MerkleTree};

pub type Merkle = MerkleTree<Sha256>;

#[cfg(test)]
mod tests {
    use rs_merkle::Hasher;

    use super::*;

    #[test]
    fn test_merkle() {
        let elems = [1, 2, 3, 4, 5, 6, 7, 8];
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
        let root_hex = merkle.root_hex().expect("Failed to get root");
        println!("Root: {}", root_hex);
        let proof = merkle.proof(&indices_to_prove);
        println!(
            "Proof: {}",
            proof
                .root_hex(&indices_to_prove, leaves_to_prove, leaves.len())
                .unwrap()
        );
    }
}
