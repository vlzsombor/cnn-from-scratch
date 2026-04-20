use ndarray::Array3;
use crate::train::activation::ReluActivation;
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::ImageData::ImageData;

#[derive(Debug)]
pub struct CnnSigmoidActivation {}
impl CnnSigmoidActivation{
    pub fn new()-> Self{
        CnnSigmoidActivation{

        }
    }
}
impl CnnLayerable for CnnSigmoidActivation {
    fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32> {
        x.image.get_sigmoid()
    }
    fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32> {
        delta_c.image.sigmoid_derivative_from_activation()
    }
}

