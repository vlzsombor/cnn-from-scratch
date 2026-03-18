use crate::train::layer::{Activation, Layer};
use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct LayerContainer {
    pub layers: Vec<Layer>
}

impl LayerContainer {

    pub fn new_def() -> Self {
        let layers = vec![
            Layer::new(2,2, Some(Activation::Relu()),None),
            Layer::new(2,1, Some(Activation::Relu()),None)
        ];
        LayerContainer {
            layers
        }
    }

    pub fn new_layers(layers: Vec<Layer>) -> Self {
        LayerContainer {
            layers
        }
    }
    pub fn new(layer_number: Vec<[u32; 2]>) -> Self {
        let layers: Vec<Layer> = layer_number.iter().map(|x|{
            Layer::new(x[0],x[1], Some(Activation::Relu()), None)
        }).collect();
        LayerContainer {
            layers
        }
    }
    #[allow(non_snake_case)]
    pub fn forward(&mut self, X: &Array2<f32>) -> Array2<f32>{
        self.layers
            .iter_mut()
            .fold(X.clone(), |acc, layer| {
                layer.forward(&acc)
            })
    }
/*
Task:
Given two binary inputs, predict their XOR output.
[0,0] → 0
[0,1] → 1
[1,0] → 1
[1,1] → 0
Network requirements:
Input layer: 2 neurons
Hidden layer: ≥2 neurons with ReLU (4 is safe)
Output layer: 1 neuron with ReLU
Loss: MSE
Full dataset is 4 samples — use all 4 as one batch every forward pass
 */
    pub fn backward_hard_coded(&mut self, loss: &Array2<f32>){
//        let a1 = col - &y;
// y_hat - y
        for l in self.layers.iter_mut().rev(){
            l.back_prop3(loss);
        }
    }
    pub fn backward2(&mut self, input: Array2<f32>)
    {
        for l in self.layers.iter_mut().rev()
        {
            let a = l.back_prop2(&input);
            let a = l.back_prop2(&input);
        }
        todo!()
    }
    //3 2 2
//    pub fn backward(&mut self, X: Array1<f32>) {
//        self.layers
//            .iter_mut()
//            .rev()
//            .fold(X, |acc, layer| {
//                layer.back_propagation(&acc)
//            });
//    }
}


pub fn loss(y_hat: &Array2<f32>, y: &Array2<f32>) -> f32 {
    1.0 / (y_hat.nrows() as f32) * ((y - y_hat) * (y - y_hat)).sum()
}

pub fn loss_derivate(y_hat: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    2.0 / (y_hat.nrows() as f32) * (y_hat - y)
}
#[cfg(test)]
mod tests {
    use crate::train::LayerContainer::{loss, loss_derivate, LayerContainer};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array2, ArrayBase, Ix2, OwnedRepr};
    use crate::train::layer::{Activation, Layer};

    #[test]
    pub fn Backprop(){
        let layers = vec![
            Layer::new(2,2, Some(Activation::Relu()),None),
            //            Layer::new(2,1, Some(Activation::Relu()),None),
        ];
        let mut sut = LayerContainer::new_layers(layers);
        let input = array![[1.,1.], [5.,5.]];
        let y = array![[1.,3.], [8.,6.],[48.,11.], [224.,19.]];
        let y = array![[2.,2.], [5., 5.]];

        let mut y_hat = sut.forward(&input);
        //        let loss: Array2<f32> = 1./(y_hat.nrows() as f32) * (y_hat - y);
        let l1 = loss(&y_hat, &y);
        for _ in 0..100{
            y_hat = sut.forward(&input);
            sut.backward_hard_coded(&loss_derivate(&y_hat, &y));
        }
        let l2 = loss(&y_hat, &y);
        dbg!(l1);
        dbg!(l2);
        assert!(l1 > l2);



    }
    #[test]
    pub fn test1()
    {
        let layer_number = vec![[3,2],[2,1]];
        let mut sut = LayerContainer::new(layer_number);
        let input: Array2<f32> = Array2::from(vec![[1.,2.,3.], [5.,6.,7.], [9.,10.,11.], [20.,300.,200000.]]);
        let input: Array2<f32> = Array2::from(vec![
            [10.,0., 10.],
            [10.,10., 10.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.]
        ]);
        let res = sut.forward(&input);
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