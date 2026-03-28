extern crate core;

use crate::util::mnist_helper::{load_mnist, train_mnist, train_mnist_cnn};

pub mod util;
pub mod train;


fn main() {
    let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

//    let (x, y) = load_mnist("C:\\Users\\ZsomborVeres-Lakos\\Downloads\\trainData.csv").unwrap();
    let accuracy = train_mnist_cnn(x,y).unwrap();
}