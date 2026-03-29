extern crate core;

use crate::train::convolutional_layer::ConvolutionalLayer;
use crate::train::layer::{ActivationLayer, Layer};
use crate::train::layer_container::LayerContainer;
use crate::util::mnist_helper::{load_mnist, train_mnist, train_mnist_cnn};

pub mod util;
pub mod train;


fn main() {
    let alpha = 0.001;
    let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
    let cnn = ConvolutionalLayer::new(Layer::new(1, 2, alpha), 28usize);
    let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
        Box::new(cnn),
        Box::new(Layer::new(120, 84, alpha)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(84, 10, alpha)),
    ];
    let mut sut = LayerContainer::new_layers_boxed(layers);
    sut.forward(&x);
//    let accuracy = train_mnist_cnn(x,y).unwrap();
    dbg!("end");
}