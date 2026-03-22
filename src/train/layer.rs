 pub(crate) use crate::train::activation::{Activation, ALPHA};
use ndarray::linalg::Dot;
use ndarray::{stack, Array, Array1, Array2, Axis, Ix1};
 use ndarray_rand::rand::SeedableRng;
 use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
 use rand_chacha::ChaCha8Rng;

 #[derive(Debug)]
pub struct Layer
{
//    number_of_inputs: u32,
//    number_of_nodes: u32,
    activation: Activation,
    /// e { features layer X features layer-1}
    weights: Array2<f32>,
    /// e { features X 1 }
    bias: Array1<f32>,
    /// raw input from the previous layer
    /// input e { batch X features }
    input: Array2<f32>,
    /// node output without activation layer
    /// z e { batch X features }
    z: Array2<f32>
}
// matrix_name e (is element of) {rows_number X column_number}
impl Layer {

    pub fn new_deterministic(number_of_inputs: u32, number_of_nodes: u32, activation: Option<Activation>, state: Option<u64>) -> Layer {
        let activation = activation.unwrap_or(Activation::empty());

        let mut rng = ChaCha8Rng::seed_from_u64(state.unwrap_or(42));
        let weights = Array2::random_using(
            (number_of_inputs as usize, number_of_nodes as usize),
            StandardNormal,
            &mut rng
        );
        //        let bias: f32 = rng.random();
        let bias: Array1<f32> =  Array1::random_using(
            number_of_nodes as usize,
            StandardNormal,
            &mut rng
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
    pub fn new(number_of_inputs: u32, number_of_nodes: u32, activation: Option<Activation>) -> Layer {
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
    /// dC_da e { b x f }
    pub fn backward_propagation(&mut self, dC_da : &Array2<f32>) -> Array2<f32>
    {
        // e { b X f }
        let da_dz = (self.activation.derivative_activation)(self.z.view());
        // e { b X f }
        let delta = dC_da * da_dz;
        let dz_db = 1.;
        let dz_dw = &self.input;

        // e { i X f }
        let dz_dal_1 = self.weights.clone();
        let b_update = (&delta * dz_db).sum_axis(Axis(0));
        self.bias = &self.bias - ALPHA * b_update;
        let w_update = &self.input.t().dot(&delta);
        self.weights = &self.weights - ALPHA * w_update;

//        dbg!(delta.shape(), dz_dal_1.shape());
        // e { b X i }
        let r = delta.dot(&dz_dal_1.t());
//        dbg!(&r.shape());
        r
    }

    fn to_array2(nested: Array<Array1<f32>, Ix1>) -> Array2<f32> {
        let views: Vec<_> = nested.iter().map(|a| a.view()).collect();
        stack(Axis(0), &views)
            .expect("Arrays must have the same length to stack into a matrix")
    }




    //    /// dL_da: e {batch X features}
    //    pub fn backward_propagation(&mut self, dL_da :Array2<f32>)
    //    {
    //        // dz_da e { batch X features }
    //        let dz_da = (self.activation.derivative_activation)(self.z.view());
    //        // dz_da e { batch X features }
    //        let delta = dL_da * dz_da;
    //        // dz_da e { batch X features_layer_-1 }
    //        let dz_dw = &self.input;
    //        let dz_db = 1.;
    //        // w_update e { batch X features-1 }
    //        let w_update = &delta * dz_dw;
    //        // e { batch X features }
    //        let partial_derivative = &delta * dz_db;
    //        let b_update = partial_derivative.mean_axis(Axis(0)).unwrap();
    //
    //        self.bias = &self.bias - b_update;
    //
    //    }

    //    pub fn backward_propagation(&mut self, dL_dact: &Array2<f32>) -> Array2<f32>
//    {
//        let dact_dz = self.z.map_axis(Axis(1), |x| {
//            (&self.activation.derivative_activation)(x.to_owned())
//        });
//        let dact_dz = Self::to_array2(dact_dz);
//        let dz_db = 1.;
//
//        let delta = dL_dact * &dact_dz;                          // (batch, n_out)
//        let b_delta = delta.mean_axis(Axis(0)).unwrap() * dz_db;          // (n_out,)
//        let w_delta = delta.t().dot(&self.input) / (delta.ncols() as f32);   // (n_out, n_in)
//        let dz_da_1 = self.weights.clone();
//        self.bias = &self.bias - ALPHA * b_delta;
//        self.weights = &self.weights - ALPHA * w_delta.t().to_owned();
//        let dL_da_prev = delta.dot(&dz_da_1.t());  // (batch, n_in)
//        dL_da_prev
//    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_neural_layer_seeded_reproducible() {
        let layer1 = Layer::new(3, 4, Some(Activation::relu()));
        assert_eq!(layer1.weights.shape(), &[3, 4]);
        let mut nnlayer1 = Layer::new_deterministic(2, 2, Some(Activation::relu()), None);
        let input: Array2<f32> = Array2::from(vec![[4.0, 2.0], [3.0, 2.0]]);
        let res = nnlayer1.forward(&input);
        let expected :Array2<f32>= array! [[0.97810096, 5.3549976], [0.50011975, 4.020927]];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }

    #[test]
    fn test_layer_stats_normal() {
        let layer = Layer::new(100, 50, Some(Activation::relu()));
        let mean = layer.weights.mean().unwrap();
        let std = layer.weights.std(0.); // population std

        // Rough checks for StandardNormal (μ=0, σ=1)
        assert!((-0.2..=0.2).contains(&mean));
        assert!((0.8..=1.2).contains(&std));
    }
}
