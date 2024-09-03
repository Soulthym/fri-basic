use crate::field::MOD;

pub trait Repr {
    fn repr(&self) -> String;
    fn dbg(&self) -> String {
        self.repr()
    }
}

pub trait ReprMod: Repr {
    fn repr_mod(&self) -> String {
        format!("{} % {}", self.repr(), MOD)
    }
}

pub trait Rand {
    type Multi;
    fn rand() -> Self;
    fn rand_n(n: u64) -> Self::Multi;
}

pub trait RandExc<T> {
    type Multi;
    fn rand_except(excluded: &Vec<T>) -> Self;
    fn rand_n_except(n: u64, excluded: &Vec<T>) -> Self::Multi;
}

pub trait DivMod<T> {
    fn divmod(&self, other: T) -> (Self, Self)
    where
        Self: Sized;
    fn div(&self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).0
    }
    fn modulus(&self, other: T) -> Self
    where
        Self: Sized,
    {
        self.divmod(other).1
    }
}

pub trait Pow<T> {
    type Output;
    fn pow(&self, n: T) -> Self::Output;
    fn pow_2_pow(&self, powlog: u64) -> Self::Output;
}

pub trait SubGroup {
    type Output;
    fn order(&self) -> u64;
    fn subgen(size: u64) -> Self::Output;
}
