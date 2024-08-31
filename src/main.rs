mod arithmetic;

fn main() {
    println!("arithmetic");
    let u: u8 = 127;
    let i: i8 = 127;
    println!("u = {}", &u);
    println!("i = {}", &i);
    println!("u + 1 = {}", &u + 1);
    println!("i + 1 = {}", i as u8 + 1);
    let u: u8 = 0;
    let i: i8 = 0;
    println!("i = {}", &i);
    println!("u = {}", &u);
    println!("i - 1 = {}", i - 1);
    println!("u - 1 = {}", u as i8 - 1);
}
