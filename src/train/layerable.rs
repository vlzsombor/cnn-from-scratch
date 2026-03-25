use core::fmt;
use ndarray::Array2;

pub trait Layerable: fmt::Debug{
    fn forward(&mut self, X: &Array2<f32>) -> Array2<f32>;
    fn backward_propagation(&mut self, dc_da: &Array2<f32>) -> Array2<f32>;
}
