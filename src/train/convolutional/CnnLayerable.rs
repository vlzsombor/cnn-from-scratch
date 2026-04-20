use core::fmt;
use ndarray::{Array2, Array3};
use crate::train::convolutional::ImageData::ImageData;

pub trait CnnLayerable: fmt::Debug{
    fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32>;
    fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32>;
}
