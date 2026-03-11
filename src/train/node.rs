use std::cmp;
use ndarray::{Array1, Array2, ArrayBase, Ix1};
use ndarray_rand::rand::{Rng, SeedableRng};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand_chacha::ChaCha8Rng; // or ChaCha20Rng, StdRng, etc.

#[derive(Debug)]
pub struct NnLayer
{
    number_of_inputs: u32,
    number_of_nodes: u32,
    activation: fn(f32) -> f32,
    weights: Array2<f32>,
    bias: f32
}

impl NnLayer{
    pub fn new(number_of_inputs: u32, number_of_nodes: u32, activation: fn(f32) -> f32, state: u64) -> NnLayer{
        let mut rng = ChaCha8Rng::seed_from_u64(state); // fixed seed
        let weights = Array2::random_using(
            (number_of_inputs as usize, number_of_nodes as usize),
            StandardNormal,
            &mut rng
        );
        print!("{}\n", weights);
        let bias: f32 = rng.random();
        NnLayer {
            number_of_inputs,
            number_of_nodes,
            activation,
            weights,
            bias
        }
    }

    pub fn forward(self, X: &Array2<f32>) -> Array1<f32>
    {
        let Xt = X.t();
        let r = Xt.dot(&self.weights);
        let array1: Array1<f32> = r
            .into_shape_with_order(self.number_of_nodes as usize)
            .expect("Matrix sizes are not in correct from");
        array1.mapv(|x| (self.activation)(x + self.bias))
    }

}


pub fn ReLU(x: f32) -> f32{
    x.max(0.0)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_layer_seeded_reproducible() {
        let layer1 = NnLayer::new(3, 4, ReLU, 42);
        assert_eq!(layer1.weights.shape(), &[3, 4]);
        let nnlayer1 = NnLayer::new(2, 2, ReLU, 42);
        let input: Array2<f32> = Array2::from(vec![[4.0], [1.0]]);
        let res = nnlayer1.forward(&input);

        println!("end res:   {}", res)
    }

    #[test]
    fn test_layer_stats_normal() {
        let layer = NnLayer::new(100, 50, ReLU,42);
        let mean = layer.weights.mean().unwrap();
        let std = layer.weights.std(0.); // population std

        // Rough checks for StandardNormal (μ=0, σ=1)
        assert!((-0.2..=0.2).contains(&mean));
        assert!((0.8..=1.2).contains(&std));
    }
}
