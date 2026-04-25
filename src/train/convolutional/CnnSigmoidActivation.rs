use ndarray::{Array3, ArrayView1};
use serde::{Deserialize, Serialize};
use crate::train::activation::ReluActivation;
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::ImageData::ImageData;

#[derive(Serialize, Deserialize, Debug)]
pub struct CnnSigmoidActivation {}
impl CnnSigmoidActivation{
    pub fn new()-> Self{
        CnnSigmoidActivation{

        }
    }
}

#[typetag::serde]
impl CnnLayerable for CnnSigmoidActivation {
    fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32> {
        x.image.get_sigmoid()
    }
    fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32> {
        delta_c.image.sigmoid_derivative_from_activation()
    }

}

