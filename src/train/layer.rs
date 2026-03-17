use std::ops::SubAssign;
use ndarray::{Array, Array1, Array2};
use ndarray::linalg::Dot;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
pub struct Layer
{
    number_of_inputs: u32,
    number_of_nodes: u32,
    activation: Activation,
    weights: Array2<f32>,
    bias: Array1<f32>,
    input: Array2<f32>
}
#[derive(Debug)]
pub struct Activation{
    activation: fn(f32) -> f32,
    derivative_activation: fn(f32) -> f32,
}
impl Activation {
    pub fn Empty() -> Self{
        Activation{
            activation: empty_activation,
            derivative_activation: |_| 1.
        }
    }
    pub fn Relu() -> Self{
        Activation{
            activation: ReLU,
            derivative_activation: Self::relu_derivative
        }
    }

    pub fn relu_derivative(x: f32) -> f32 {
        if x < 0. { 1.0 } else { 0.0 }
    }
}
trait LayerTrait {
    fn activation(&self, input: f32) -> f32;
    fn backpropagation(&self);
}

impl Layer {
    pub fn new(number_of_inputs: u32, number_of_nodes: u32, activation: Option<Activation>, state: Option<u64>) -> Layer {
        let activation = activation.unwrap_or(Activation::Empty());

        let state = state.unwrap_or(42);
        let mut rng = ChaCha8Rng::seed_from_u64(state); // fixed seed
        let weights = Array2::random_using(
            (number_of_inputs as usize, number_of_nodes as usize),
            StandardNormal,
            &mut rng
        );
//        let bias: f32 = rng.random();
        let bias: Array1<f32> =  Array1::random_using(
            (number_of_nodes as usize),
            StandardNormal,
            &mut rng
        );
        Layer {
            number_of_inputs,
            number_of_nodes,
            activation,
            weights,
            bias,
            input: Array2::zeros((number_of_inputs as usize, number_of_nodes as usize))
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&mut self, X: &Array2<f32>) -> Array2<f32>
    {
        let r = X.dot(&self.weights);
        self.input = X.clone();

//        let mut i =0;
//        r.mapv(|x| {
//            let ret = (self.activation.activation)(x + self.bias[i]);
//            i = i + 1;
//            return ret;
//        });
        (&r + &self.bias).mapv(|x| (self.activation.activation)(x))
    }
    pub fn back_propagation(&mut self, grad_output: &Array1<f32>) -> Array1<f32>
    {
        let dact_dz = grad_output.mapv(|x|(self.activation.derivative_activation)(x));
        let dz_dw:Array2<f32> = &dact_dz * &self.input;
        let dz_db = &dact_dz;
        let dz_dactPrevious = self.weights.clone();
        self.weights = &self.weights - dz_dw;
        self.bias = &self.bias - dz_db;

        dact_dz.dot(&dz_dactPrevious)
    }
}
#[allow(non_snake_case)]
pub fn ReLU(x: f32) -> f32{
    x.max(0.0)
}

pub fn empty_activation(x: f32) -> f32
{
    x
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_neural_layer_seeded_reproducible() {
        let layer1 = Layer::new(3, 4, Some(Activation::Relu()), Some(42));
        assert_eq!(layer1.weights.shape(), &[3, 4]);
        let mut nnlayer1 = Layer::new(2, 2, Some(Activation::Relu()), Some(42));
        let input: Array2<f32> = Array2::from(vec![[4.0, 2.0], [3.0, 2.0]]);
        let res = nnlayer1.forward(&input);
        let expected :Array2<f32>= array! [[2.2273476, 7.026132], [1.7493664, 5.6920614]];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }

    #[test]
    fn test_layer_stats_normal() {
        let layer = Layer::new(100, 50, Some(Activation::Relu()), Some(42));
        let mean = layer.weights.mean().unwrap();
        let std = layer.weights.std(0.); // population std

        // Rough checks for StandardNormal (μ=0, σ=1)
        assert!((-0.2..=0.2).contains(&mean));
        assert!((0.8..=1.2).contains(&std));
    }
}
