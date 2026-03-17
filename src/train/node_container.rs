use crate::train::layer::{Activation, Layer};
use ndarray::{Array1, Array2};

struct NodeContainer {
    layers: Vec<Layer>
}

impl NodeContainer {

    pub fn new_def() -> Self {
        let layers = vec![
            Layer::new(3,2, Some(Activation::Relu()),None),
            Layer::new(2,2, Some(Activation::Relu()),None),
            Layer::new(2,2, None,None)
        ];
        NodeContainer {
            layers
        }
    }
    pub fn new(layer_number: Vec<[u32; 2]>) -> Self {
        let layers: Vec<Layer> = layer_number.iter().map(|x|{
            Layer::new(x[0],x[1], Some(Activation::Relu()), None)
        }).collect();
        NodeContainer {
            layers
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&mut self, X: Array2<f32>) -> Array2<f32>{
        self.layers
            .iter_mut()
            .fold(X, |acc, layer| {
                layer.forward(&acc)
            })
    }
    //3 2 2
    pub fn backward_def(&mut self, X: Array1<f32>) {
        self.layers
            .iter_mut()
            .fold(X, |acc, layer| {
                layer.back_propagation(&acc)
            });
    }
}


#[cfg(test)]
mod tests {
    use crate::train::node_container::NodeContainer;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2};

    #[test]
    pub fn test1()
    {
        let layer_number = vec![[3,2],[2,1]];
        let mut sut = NodeContainer::new(layer_number);
        let input: Array2<f32> = Array2::from(vec![[1.,2.,3.], [5.,6.,7.], [9.,10.,11.], [20.,300.,200000.]]);
        let input: Array2<f32> = Array2::from(vec![
            [10.,0., 10.],
            [10.,10., 10.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.]
        ]);
        let res = sut.forward(input);
        let expected :Array2<f32>= array![
            [6.529471],
            [12.884275],
            [59731.12],
            [59731.12],
            [59731.12]
        ];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }
}