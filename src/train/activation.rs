use crate::train::loss_functions::softmax;
use ndarray::{Array2, ArrayView2};

#[derive(Debug)]
pub struct Activation{
    pub activation: fn(ArrayView2<f32>) -> Array2<f32>,
    pub derivative_activation: fn(ArrayView2<f32>) -> Array2<f32>,
}
impl Activation {
    pub fn relu() -> Self{
        Activation{
            activation: Self::ReLU,
            derivative_activation: Self::ReLU_derivative,
        }
    }
    pub fn softmax() -> Self{
        Activation{
            activation: softmax,
            derivative_activation: |x| Array2::ones(x.raw_dim())
        }
    }
    #[allow(non_snake_case)]
    fn ReLU_derivative(x: ArrayView2<f32>) -> Array2<f32>
    {
        x.mapv(|xi| if xi > 0. { 1.0 } else { 0.0 })
    }

    #[allow(non_snake_case)]
    fn ReLU(x: ArrayView2<f32>) -> Array2<f32>
    {
        x.mapv(|xi| {xi.max(0.0)})
    }
}
