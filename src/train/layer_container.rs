use crate::train::layer::Layer;
use crate::train::layerable::Layerable;
use crate::train::loss_functions::{softmax, EPSILON};
use ndarray::Array2;
#[derive(Debug)]
pub struct LayerContainer {
    pub layers: Vec<Box<dyn Layerable>>
}

impl LayerContainer {

    pub fn new_def() -> Self {
        let layers: Vec<Box<dyn Layerable>> = vec![
            Box::new(Layer::new(2, 2, 0.001)),
            Box::new(Layer::new(2, 1, 0.001))
        ];

        LayerContainer {
            layers
        }
    }

    pub fn new_layers_boxed(layers: Vec<Box<dyn crate::train::layerable::Layerable>>) -> Self {
        LayerContainer {
            layers
        }
    }
    pub fn new_layers(layers: Vec<Layer>) -> Self {
        LayerContainer {
            layers: layers.into_iter().map(|l| Box::new(l) as Box<dyn Layerable>).collect()
        }
    }

    pub fn new(layer_number: Vec<[u32; 2]>, alpha: f32) -> Self {
        let layers: Vec<Layer> = layer_number.iter().map(|x|{
            Layer::new(x[0], x[1], alpha)
        }).collect();
        LayerContainer {
            layers: layers.into_iter().map(|l| Box::new(l) as Box<dyn Layerable>).collect()
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
    pub fn backward_propagation(&mut self, dc_da: Array2<f32>) {
        self.layers
            .iter_mut()
            .rev()
            .fold(dc_da, |acc, item|{
            item.backward_propagation(&acc)
        });
    }
}


pub fn loss(y_hat: &Array2<f32>, y: &Array2<f32>) -> f32 {
    1.0 / (y_hat.nrows() as f32) * ((y - y_hat) * (y - y_hat)).sum()
}

pub fn mse_loss_derivative(y_hat: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    2.0 / (y_hat.nrows() as f32) * (y_hat - y)
}

pub fn cross_entropy_loss_and_softmax(y_hat: &Array2<f32>, y: &Array2<f32>) -> Array2<f32> {
    let softmax = softmax(y_hat.view());
    softmax - y
}

pub fn cross_entropy_loss(y_hat: &Array2<f32>, y: &Array2<f32>) -> f32 {
    let _batch = y_hat.shape()[0] as f32;
    let loss = -( y * y_hat.mapv(|x| (x+EPSILON).ln()) ).sum();
    loss
}
#[cfg(test)]
mod tests {
    use crate::train::layer::{ActivationLayer, Layer};
    use crate::train::layer_container::{cross_entropy_loss_and_softmax, loss, mse_loss_derivative, LayerContainer};
    use crate::util::util::{accuracy, debug_array, normalize_features};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2, Axis};
    use std::error::Error;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use ndarray_rand::RandomExt;
    use rand_distr::StandardNormal;

    fn generate_xor_dataset() -> (Array2<f32>, Array2<f32>) {
        let x: Array2<f32> = array![[0.,0.],
                                [0.,1.],
                                [1.,0.],
                                [1.,1.],
    ];
        // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
        let y = array![[0., 1.], [1., 0.], [1., 0.],[0., 1.]];
        (x, y)
    }
    fn generate_linear_dataset(n_samples: usize) -> (Array2<f32>, Array2<f32>) {
        let x: Array2<f32> = Array2::random((n_samples, 2), StandardNormal) * 2. + 10.;
        // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
        let y: Array1<f32> = x.column(0).to_owned() + x.column(0).to_owned();
        (x, y.insert_axis(Axis(1))
            .to_owned())
    }
    #[test]
    pub fn multi_layer(){
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new(2, 4, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(4, 2, 0.001)),
        ];
        let mut sut = LayerContainer::new_layers_boxed(layers);
        let input = array![[1., 1.], [5.,5.], [7.,7.], [10., 10.], [12., 12.]];
        let y = array![[2., 2.], [10., 10.], [14., 14.], [20.,20.], [24., 24.]];

        let mut y_hat = sut.forward(&input);
        let l1 = loss(&y_hat, &y);
        for _ in 0..1000{
            y_hat = sut.forward(&input);
            sut.backward_propagation(mse_loss_derivative(&y_hat, &y));
        }
        let l2 = loss(&y_hat, &y);
        dbg!(l1);
        dbg!(l2);
        assert!(l1 > l2);
    }
    #[test]
    pub fn test_backprop_decrease_loss2(){
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new(1, 2, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(2, 1, 0.001)),
        ];
        let mut sut = LayerContainer::new_layers_boxed(layers);
        let input = array![[1.], [5.,]];
        let y = array![[2.], [10.]];

        let mut y_hat = sut.forward(&input);
        //        let loss: Array2<f32> = 1./(y_hat.nrows() as f32) * (y_hat - y);
        let l1 = loss(&y_hat, &y);
        for _ in 0..100{
            y_hat = sut.forward(&input);
            sut.backward_propagation(mse_loss_derivative(&y_hat, &y));
        }
        let l2 = loss(&y_hat, &y);
        dbg!(l1);
        dbg!(l2);
        assert!(l1 > l2);
    }
    #[test]
    pub fn test_backprop_decrease_loss(){
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new(2, 2, 0.001)),
        ];
        let mut sut = LayerContainer::new_layers_boxed(layers);
        let input = array![[1.,1.], [5.,5.]];
        let y = array![[2.,2.], [10., 10.]];

        let mut y_hat = sut.forward(&input);
        //        let loss: Array2<f32> = 1./(y_hat.nrows() as f32) * (y_hat - y);
        let l1 = loss(&y_hat, &y);
        for _ in 0..100{
            y_hat = sut.forward(&input);
            sut.backward_propagation(mse_loss_derivative(&y_hat, &y));
        }
        let l2 = loss(&y_hat, &y);
        dbg!(l1);
        dbg!(l2);
        assert!(l1 > l2);
    }
    #[test]
    pub fn xor()
    {
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new(2, 64, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(64, 64, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(64, 2, 0.001)),
        ];
        let mut sut = LayerContainer::new_layers_boxed(layers);

        let (x, y) = generate_xor_dataset();

        let mut y_hat = sut.forward(&x);
        let l1 = loss(&y_hat, &y);
        for i in 0..10000{
            y_hat = sut.forward(&x);
            sut.backward_propagation(mse_loss_derivative(&y_hat, &y));
            if i % 1000 == 0 {
                let _ = loss(&y_hat, &y);
            }
        }
        let l2 = loss(&y_hat, &y);

        let xp = array![[0.,0.], [0., 1.], [1.,0.], [1.,1.]];
        let predict1 = sut.forward(&xp);
        debug_array(&xp);
        debug_array(&predict1);
        assert!(l1 > l2);
    }
    #[test]
    pub fn test1()
    {
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new_deterministic(3, 2, None, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new_deterministic(2, 1, None, 0.001)),
            Box::new(ActivationLayer::relu()),
        ];

        let mut sut = LayerContainer::new_layers_boxed(layers);
        let input: Array2<f32> = Array2::from(vec![
            [10.,0., 10.],
            [10.,10., 10.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.],
            [100000.,10000., 100000.]
        ]);
        let res = sut.forward(&input);
        let expected :Array2<f32>= array![[6.349539],
            [12.704343],
            [59730.938],
            [59730.938],
            [59730.938]
        ];
        assert_abs_diff_eq!(&res, &expected, epsilon = 1e-4);
    }

    #[test]
    pub fn iris_dataset_test()
    {
        let r = load_iris("src/data/Iris.csv").unwrap();
        let result = train_iris(&r);
        assert!(result.is_some());

        assert!(result.unwrap() > 0.9);
    }
    fn train_iris(data: &Vec<(Vec<f32>, String)>) -> Option<f32>
    {
        let x: Vec<f32> = data.iter()
            .map(|(x, _)| x)
            .flatten()
            .copied()
            .collect();

        let x: Array2<f32> = Array2::from_shape_vec((150, &x.len()/150), x).unwrap();
        let x = normalize_features(&x);
        let label: Vec<Vec<f32>> = data
            .iter()
            .map(|(_, y)| {
                if y == "Iris-setosa" { vec![1.0, 0.0, 0.0] } else if y == "Iris-versicolor" { vec![0.0, 1.0, 0.0] } else { vec![0.0, 0.0, 1.0]}
            })
            .collect();

        let flat: Vec<f32> = label.iter().flatten().cloned().collect();
        let y = Array2::from_shape_vec((label.len(), label[0].len()), flat).unwrap();
        //    let y :Array2<f32> = Array2::from_shape_vec((150,1), label).unwrap();
        let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
            Box::new(Layer::new(4, 32, 0.001)),
            //        Box::new(ActivationLayer::relu()),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(32, 64, 0.001)),
            Box::new(ActivationLayer::relu()),
            Box::new(Layer::new(64, 3, 0.001)),
            //        Box::new(ActivationLayer::softmax_with_cross_entropy_loss()),
        ];
        let mut sut = LayerContainer::new_layers_boxed(layers);
        for i in 0..100{
            let y_hat = sut.forward(&x);
            sut.backward_propagation(cross_entropy_loss_and_softmax(&y_hat, &y));
            if i < 20  {
                let accuracy = accuracy(&y_hat, &y);
                dbg!(accuracy);
            }
        }
        let y_hat = sut.forward(&x);
        let accuracy = accuracy(&y_hat, &y);
        Some(accuracy)
    }
    fn load_iris(path: &str) -> Result<Vec<(Vec<f32>, String)>, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut data = Vec::new();

        for line in reader.lines().skip(1) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() == 6 {
                let class = parts[5];
                let features: Vec<f32> = parts[1..5]
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();

                data.push((features, class.to_string()));
            }
        }
        if data.is_empty() {
            return Err("Keine Daten gefunden".into());
        }
        Ok(data)
    }

}