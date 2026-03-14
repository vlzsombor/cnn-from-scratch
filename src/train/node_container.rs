use crate::train::layer::Layer;
use ndarray::Array2;

struct NodeContainer {
    layers: Vec<Layer>
}

impl NodeContainer {

    pub fn new(layer_number: Vec<[u32; 2]>) -> Self {
        let layers: Vec<Layer> = layer_number.iter().map(|x|{
            let a = x[0];
            let b = x[1];
            Layer::new(a,b, None, None)
        }).collect();
        NodeContainer {
            layers
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&self, X: Array2<f32>) -> Array2<f32>{
//        self.layers
//            .iter()
//            .for_each(|x|{
//                x.forward()
//            });
        self.layers
            .iter()
            .fold(X, |acc, layer| {
                layer.forward(&acc)
            })
    }
}


#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use crate::train::node_container::NodeContainer;
    use ndarray::{array, Array2};

    #[test]
    pub fn test1()
    {
        let layer_number = vec![[3,2],[2,1]];
        let sut = NodeContainer::new(layer_number);
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