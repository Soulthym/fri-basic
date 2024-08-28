use std::cmp::max;
use std::ops::{Add, Div, Fn, Mul, Rem, Sub};

use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use ark_ff::UniformRand;
use ark_poly::univariate::DensePolynomial;
use ark_poly::DenseUVPolynomial;

use rand::rngs::ThreadRng;
use rand::thread_rng;

trait Print {
    fn print(&self);
    fn println(&self);
    fn print_named(&self, name: &str);
    fn println_named(&self, name: &str);
}

#[derive(MontConfig)]
#[modulus = "5"] // 3221225473 = 3 * 2.pow(30) + 1
#[generator = "2"]
pub struct FConf;
pub type Fq = Fp64<MontBackend<FConf, 1>>;

impl Print for Fq {
    fn println(&self) {
        self.print();
        println!();
    }
    fn print(&self) {
        print!("{} % {}", self, FConf::MODULUS);
    }
    fn println_named(&self, name: &str) {
        self.print_named(name);
        println!();
    }
    fn print_named(&self, name: &str) {
        print!("{}(x) = ", name);
        self.print();
    }
}

type PolyBase = DensePolynomial<Fq>;
#[derive(Debug, Clone)]
struct Poly {
    poly: PolyBase,
    roots: Vec<Fq>,
}

impl Poly {
    fn new() -> Self {
        Poly {
            poly: DensePolynomial { coeffs: vec![] },
            roots: vec![],
        }
    }

    fn from(value: Vec<Fq>) -> Self {
        let mut res = Poly::new();
        res.poly.coeffs.extend(value);
        res
    }

    fn rand(rng: &mut ThreadRng) -> Self {
        let mut poly = PolyBase::rand(0, rng);
        while poly[0] == Fq::from(0) {
            poly = PolyBase::rand(0, rng);
        }
        let root = poly[0];
        Poly {
            poly,
            roots: vec![root],
        }
    }

    fn rand_n(n: usize, rng: &mut ThreadRng) -> Self {
        let mut res = Poly::from(vec![Fq::from(1)]);
        let mut roots = vec![-res.poly[0]];
        res.println_named("res");
        let x: Poly = Poly::from(vec![Fq::from(0), Fq::from(1)]);
        x.println_named("x");
        for _ in 0..n {
            let root = Poly::rand(rng);
            root.println_named("root");
            res = (x.clone() - root.clone()) * res.clone();
            res.println_named("new");
            roots.push(root.poly[0]);
        }
        Poly {
            poly: res.poly,
            roots,
        }
    }

    fn print_roots(&self) {
        if self.roots.is_empty() {
            self.println();
            print!("are unknown");
        } else {
            for root in self.roots.clone() {
                print!("(x - {})", root);
            }
        }
    }

    fn println_roots(&self) {
        self.print_roots();
        println!();
    }

    fn print_named_roots(&self, name: &str) {
        print!("{}(x) = ", name);
        self.print_roots();
    }

    fn println_named_roots(&self, name: &str) {
        self.print_named_roots(name);
        println!();
    }
}

impl Mul<Poly> for Poly {
    type Output = Self;
    fn mul(self, other: Poly) -> Self::Output {
        let len = self.poly.coeffs.len() + other.poly.coeffs.len() + 1;
        let mut coeffs = vec![Fq::from(0); len];
        //self.print_named("a");
        //println!("*");
        //other.print_named("b");
        for (l_exp, left) in self.poly.coeffs.iter().enumerate() {
            for (r_exp, right) in other.poly.coeffs.iter().enumerate() {
                coeffs[l_exp + r_exp] += left * right;
            }
        }
        let poly = Poly::from(coeffs);
        //poly.print_named("(a * b)");
        poly
    }
}

impl Sub<Poly> for Poly {
    type Output = Self;
    fn sub(self, other: Poly) -> Self::Output {
        let mut poly = Poly::new();
        //println!("{:?} - {:?}", self.poly.coeffs, other.poly.coeffs);
        let (left_len, right_len) = (self.poly.coeffs.len(), other.poly.coeffs.len());
        let len = max(left_len, right_len);
        //println!("#{}", len);
        for i in 0..len {
            let left = if i < left_len {
                self.poly.coeffs[i]
            } else {
                Fq::from(0)
            };
            let right = if i < right_len {
                other.poly.coeffs[i]
            } else {
                Fq::from(0)
            };
            poly.poly.coeffs.push(left - right);
        }
        //poly.print_named("(a - b)");
        poly
    }
}

impl Print for Poly {
    fn println(&self) {
        self.print();
        println!();
    }
    fn print(&self) {
        let mut first = true;
        for (exp, coef) in self.poly.coeffs.iter().enumerate().rev() {
            if *coef == Fq::from(0) {
                continue;
            }
            let prefix = if first { "" } else { " + " };
            let rcoef = if *coef != Fq::from(1) {
                &format!("{}", coef)
            } else {
                ""
            };
            match exp {
                0 => print!("{}{}", prefix, coef),
                1 => print!("{}{}x", prefix, rcoef,),
                _ => print!("{}{}x^{}", prefix, rcoef, exp),
            }
            first = false;
        }
        print!(" % {}", FConf::MODULUS);
    }
    fn print_named(&self, name: &str) {
        print!("{}(x) = ", name);
        self.print();
    }
    fn println_named(&self, name: &str) {
        self.print_named(name);
        println!();
    }
}

fn main() {
    let mut rng = thread_rng();
    let poly: Poly = Poly::rand_n(5, &mut rng);
    poly.println_named("p");
    poly.println_named_roots("p")
}
