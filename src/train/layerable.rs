use core::fmt;
use ndarray::{Array1, Array2, ArrayView1};

#[typetag::serde]
pub trait Layerable: fmt::Debug{
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32>;
    fn backward_propagation(&mut self, dc_da: &Array2<f32>) -> Array2<f32>;
}
