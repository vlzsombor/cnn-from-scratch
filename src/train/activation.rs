use crate::train::loss_functions::softmax;
use ndarray::{Array, Array2, ArrayBase, ArrayView2, DataMut, Dimension};
use ndarray_rand::rand_distr::num_traits::Float;

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

    pub fn Relu_scalar(x: f32) -> f32{
        x.max(0.0)
    }

}
pub trait ReluActivation {
    type Output;
    fn get_relu(&self) -> Self::Output;
    fn get_sigmoid(&self) -> Self::Output;
    fn get_sigmoid_derivative(&self) -> Self::Output;
    fn sigmoid_derivative_from_activation(&self) -> Self::Output;
}
impl<D, S, A> ReluActivation for ArrayBase<S, D>
where
    D: Dimension,
    S: DataMut<Elem = A>,
    A: Float
{
    type Output = Array<A, D>;
    fn get_relu(&self) -> Array<A,D>{
        self.map(|x| x.max(A::zero()))
    }
    fn get_sigmoid(&self) -> Array<A, D> {
        self.map(|x| {
            let one = A::one();
            // clamp requires PartialOrd bound on A
            let x_clamped = x.clamp(-A::from(88.0).unwrap(), A::from(88.0).unwrap());
            one / (one + (-x_clamped).exp())
        })
    }
    fn get_sigmoid_derivative(&self) -> Array<A, D>{
        self.map(|x| {
            let s = {
                let one = A::one();
                one / (one + (-*x).exp())
            };
            s * (A::one() -s)
        })
    }
    fn sigmoid_derivative_from_activation(&self) -> Array<A, D> {
        self.mapv(|s| s * (A::one() - s))
    }
}

pub fn relu_generic<D,S>(array: &ArrayBase<S, D>) -> Array<f32, D>
where
    D: Dimension,
    S: DataMut<Elem = f32>
{
    array.mapv(|e| e.max(0.0))
}
