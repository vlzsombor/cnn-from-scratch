use crate::train::activation::ReluActivation;
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::ImageData::ImageData;
use ndarray::{Array3, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CnnSigmoidActivation {
    cached_activation: Option<Array3<f32>>,
}
impl CnnSigmoidActivation {
    pub fn new() -> Self {
        CnnSigmoidActivation {
            cached_activation: None,
        }
    }
}

// #[typetag::serde]
// impl CnnLayerable for CnnSigmoidActivation {
//     fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32> {
//         x.image.get_sigmoid()
//     }
//     fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32> {
//         delta_c.image.sigmoid_derivative_from_activation()
//     }
// }

#[typetag::serde]
impl CnnLayerable for CnnSigmoidActivation {
    fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32> {
        let activated = x.image.get_sigmoid();
        self.cached_activation = Some(activated.clone()); // ← cachen
        activated
    }
    fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32> {
        let activation = self.cached_activation.as_ref().unwrap();
        activation.sigmoid_derivative_from_activation() * &delta_c.image
    }
}
