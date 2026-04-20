
pub mod util;
pub mod train;


fn main() {
    let a = vec![1,2,3,4];
    let r = a.iter().fold(100, |acc, e|{
        e+acc
    });
    println!("{}", r);
    0;
}