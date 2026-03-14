use ndarray::Array2;
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand_chacha::ChaCha8Rng;
// or ChaCha20Rng, StdRng, etc.

#[derive(Debug)]
pub struct Layer
{
    number_of_inputs: u32,
    number_of_nodes: u32,
    activation: fn(f32) -> f32,
    weights: Array2<f32>,
    bias: f32
}

impl Layer {
    pub fn new(number_of_inputs: u32, number_of_nodes: u32, activation: Option<fn(f32) -> f32>, state: Option<u64>) -> Layer {
        let activation = activation.unwrap_or(_ReLU);
        let state = state.unwrap_or(42);
        let mut rng = ChaCha8Rng::seed_from_u64(state); // fixed seed
        let weights = Array2::random_using(
            (number_of_inputs as usize, number_of_nodes as usize),
            StandardNormal,
            &mut rng
        );
        let bias: f32 = rng.random();
        Layer {
            number_of_inputs,
            number_of_nodes,
            activation,
            weights,
            bias
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&self, X: &Array2<f32>) -> Array2<f32>
    {
        let r = X.dot(&self.weights);
        r.mapv(|x| (self.activation)(x + self.bias))
    }

}
#[allow(non_snake_case)]
pub fn _ReLU(x: f32) -> f32{
    x.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_neural_layer_seeded_reproducible() {
        let layer1 = Layer::new(3, 4, Some(_ReLU), Some(42));
        assert_eq!(layer1.weights.shape(), &[3, 4]);
        let nnlayer1 = Layer::new(2, 2, Some(_ReLU), Some(42));
        let input: Array2<f32> = Array2::from(vec![[4.0, 2.0], [3.0, 2.0]]);
        let res = nnlayer1.forward(&input);
        let expected :Array2<f32>= array! [[2.2273476, 7.026132], [1.7493664, 5.6920614]];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }

    #[test]
    fn test_layer_stats_normal() {
        let layer = Layer::new(100, 50, Some(_ReLU), Some(42));
        let mean = layer.weights.mean().unwrap();
        let std = layer.weights.std(0.); // population std

        // Rough checks for StandardNormal (μ=0, σ=1)
        assert!((-0.2..=0.2).contains(&mean));
        assert!((0.8..=1.2).contains(&std));
    }
}
