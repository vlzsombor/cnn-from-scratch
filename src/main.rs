use ndarray::{array, Array2};
use crate::input_transform::{normalize_image_pixels_vec, process_csv};
use crate::train::node::{NnLayer, ReLU};

mod input_transform;
pub mod train;

const U8_UPPER: u32 = 255;

fn main() {
    println!("Hello, world!");
    let r = process_csv("src/data/mnist_train_small.csv").unwrap();
    let normalized = normalize_image_pixels_vec(&r, U8_UPPER);

    let nnlayer1 = NnLayer::new(2, 2, ReLU, 42);
    let a: Array2<f32> = Array2::from(vec![[4.0], [1.0]]);

    let res = nnlayer1.forward(&a);
    print!("result: {}\n", res);
}
