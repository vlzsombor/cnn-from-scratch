 pub(crate) use crate::train::activation::{Activation, ALPHA};
use ndarray::linalg::Dot;
use ndarray::{stack, Array, Array1, Array2, Axis, Ix1};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct Layer
{
//    number_of_inputs: u32,
//    number_of_nodes: u32,
    activation: Activation,
    weights: Array2<f32>,
    bias: Array1<f32>,
    input: Array2<f32>,
    z: Array2<f32>
}

impl Layer {
    pub fn new(number_of_inputs: u32, number_of_nodes: u32, activation: Option<Activation>, state: Option<u64>) -> Layer {
        let activation = activation.unwrap_or(Activation::empty());

//        let mut rng = ChaCha8Rng::seed_from_u64(state); // fixed seed
        let weights = Array2::random(
            (number_of_inputs as usize, number_of_nodes as usize),
            StandardNormal,
//            &mut rng
        );
//        let bias: f32 = rng.random();
        let bias: Array1<f32> =  Array1::random(
            number_of_nodes as usize,
            StandardNormal,
            //            &mut rng
        );
        Layer {
//            number_of_inputs,
//            number_of_nodes,
            activation,
            weights,
            bias,
            input: Array2::zeros((number_of_inputs as usize, number_of_nodes as usize)),
            z: Array2::zeros((number_of_inputs as usize, number_of_nodes as usize))
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&mut self, X: &Array2<f32>) -> Array2<f32>
    {
        let r = X.dot(&self.weights);
        self.input = X.clone();
        let a = &r + &self.bias ;
        let bb = a.map_axis(Axis(1), |row| (self.activation.activation)(row.to_owned()));
        let z = Self::to_array2(bb);
        self.z = z.clone();
        z
    }
    fn to_array2(nested: Array<Array1<f32>, Ix1>) -> Array2<f32> {
        // Erstellt eine Vec von Views, da stack Referenzen benötigt
        let views: Vec<_> = nested.iter().map(|a| a.view()).collect();
        // Stapelt die 1D-Arrays entlang Axis(0), um Zeilen einer 2D-Matrix zu bilden
        stack(Axis(0), &views)
            .expect("Arrays must have the same length to stack into a matrix")
    }
    /// dL_dact: (batch, n_out)
    pub fn back_prop3(&mut self, dL_dact: &Array2<f32>) -> Array2<f32>
    {
        let dact_dz = self.z.map_axis(Axis(1), |x| {
            (&self.activation.derivative_activation)(x.to_owned())
        });
        let dact_dz = Self::to_array2(dact_dz);
        let dz_db = 1.;

        let delta = dL_dact * &dact_dz;                          // (batch, n_out)
        let b_delta = delta.mean_axis(Axis(0)).unwrap() * dz_db;          // (n_out,)
        let w_delta = delta.t().dot(&self.input) / (delta.ncols() as f32);   // (n_out, n_in)
        let dz_da_1 = self.weights.clone();
        self.bias = &self.bias - ALPHA * b_delta;
        self.weights = &self.weights - ALPHA * w_delta.t().to_owned();
        let dL_da_prev = delta.dot(&dz_da_1.t());  // (batch, n_in)
        dL_da_prev
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_neural_layer_seeded_reproducible() {
        let layer1 = Layer::new(3, 4, Some(Activation::relu()), Some(42));
        assert_eq!(layer1.weights.shape(), &[3, 4]);
        let mut nnlayer1 = Layer::new(2, 2, Some(Activation::relu()), Some(42));
        let input: Array2<f32> = Array2::from(vec![[4.0, 2.0], [3.0, 2.0]]);
        let res = nnlayer1.forward(&input);
        let expected :Array2<f32>= array! [[2.2273476, 7.026132], [1.7493664, 5.6920614]];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }

    #[test]
    fn test_layer_stats_normal() {
        let layer = Layer::new(100, 50, Some(Activation::relu()), Some(42));
        let mean = layer.weights.mean().unwrap();
        let std = layer.weights.std(0.); // population std

        // Rough checks for StandardNormal (μ=0, σ=1)
        assert!((-0.2..=0.2).contains(&mean));
        assert!((0.8..=1.2).contains(&std));
    }
}
