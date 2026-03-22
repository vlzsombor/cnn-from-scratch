use crate::train::loss_functions::softmax;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug)]
pub struct Activation{
    pub activation: fn(Array1<f32>) -> Array1<f32>,
//    pub derivative_activation: fn(ArrayView1<f32>) -> Array1<f32>,
    pub derivative_activation: fn(ArrayView2<f32>) -> Array2<f32>,
}
pub const ALPHA: f32 = 0.001;
impl Activation {
    pub fn empty() -> Self{
        Activation{
            activation: Self::empty_fn,
            derivative_activation: |x| {
                Array2::ones(x.raw_dim())
            },
        }
    }

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
       x.mapv(|xi| if xi < 0. { 0. } else { 1.0 })
    }
    #[allow(non_snake_case)]
    fn ReLU(x: Array1<f32>) -> Array1<f32>
    {
        x.mapv(|xi| xi.max(0.0))
    }

    fn empty_fn(x: Array1<f32>) -> Array1<f32>{
        x
    }
}
