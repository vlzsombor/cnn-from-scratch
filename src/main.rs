extern crate core;

use ndarray::{array, s, Array3, Array4, ArrayView2};
use crate::train::convolutional_layer::ConvolutionalLayer;
use crate::train::convolutional_matlab::{ConvolutionalMatlab, ImageData};
use crate::train::layer::{ActivationLayer, Layer};
use crate::train::layer_container::LayerContainer;
use crate::util::mnist_helper::{load_mnist, train_mnist, train_mnist_cnn};

pub mod util;
pub mod train;


fn main() {

}