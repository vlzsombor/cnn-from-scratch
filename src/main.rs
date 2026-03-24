extern crate core;

use crate::train::layer::{Activation, ActivationLayer, Layer};
use crate::train::layer_container::{cross_entropy_loss, loss, mse_loss_derivative, cross_entropy_loss_and_softmax, LayerContainer};
use ndarray::{array, Array, Array1, Array2, Axis, Shape};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::train::loss_functions::softmax;
use crate::util::mnist_helper::{load_mnist, train_mnist};

pub mod util;
mod input_transform;
pub mod train;

fn generate_xor_dataset() -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = array![[0.,0.],
                                [0.,1.],
                                [1.,0.],
                                [1.,1.],
    ];
    // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y = array![[0., 1.], [1., 0.], [1., 0.],[0., 1.]];
    (x, y)
}
fn generate_linear_dataset(n_samples: usize) -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = Array2::random((n_samples, 2), StandardNormal) * 2. + 10.;
   // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y: Array1<f32> = x.column(0).to_owned() + x.column(0).to_owned();
    (x, y.insert_axis(Axis(1))
        .to_owned())
}



fn main() {
    let (X, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
    let accuracy = train_mnist(X,y).unwrap();
    dbg!(&accuracy);
}