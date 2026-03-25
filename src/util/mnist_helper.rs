use crate::train::layer::{ActivationLayer, Layer};
use crate::train::layer_container::{cross_entropy_loss_and_softmax, LayerContainer};
use crate::train::loss_functions::softmax;
use crate::util::util::{accuracy, one_hot};
use ndarray::{s, Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_mnist(path: &str) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

//    let mut data = Vec::new();

    let mut X = Vec::new();
    let mut y = Vec::new();
    for line in reader.lines(){
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        let class = parts[0];
        let features: Vec<f32> = parts[1..]
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        X.push(features);
        y.push(class.parse::<f32>().unwrap());
        //            data.push((features, class.to_string()));
    }

    let rows = X.len();
    let cols = X[0].len();
    let flat: Vec<f32> = X.into_iter().flatten().collect();
    let x_array2: Array2<f32> = Array2::from_shape_vec((rows, cols), flat).unwrap();
    let y_array: Array1<f32> = Array1::from_vec(y);
    Ok((x_array2, y_array))
}
pub fn normalize_mnist(x: &Array2<f32>) -> Array2<f32> {
    x / 255.
}
pub fn train_mnist(X: Array2<f32>, y: Array1<f32>) -> Option<f32>
{
    let subslice = 1_000;
    let X_train = normalize_mnist(&X).slice(s![..subslice, ..]).to_owned();
    let slice: Vec<usize> = y.iter().map(|&x| x as usize).collect();
    let y_train = one_hot(&slice, 10).slice(s![..subslice, ..]).to_owned();
    let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![

        Box::new(Layer::new(28*28, 128, 0.0001)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(128, 128,  0.0001)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(128, 10, 0.0001)),
        //        Box::new(ActivationLayer::softmax_with_cross_entropy_loss()),
    ];
    let mut sut = LayerContainer::new_layers_boxed(layers);
    for i in 0..500{
        let y_hat = sut.forward(&X_train);
        sut.backward_propagation(cross_entropy_loss_and_softmax(&y_hat, &y_train));
        if i % 100 == 0  {
            let accuracy = accuracy(&y_hat, &y_train);
            dbg!(i, accuracy);
        }
    }
    let y_hat = sut.forward(&X_train);

    dbg!(softmax(y_hat.view()).row(0));
    let X_test = normalize_mnist(&X).slice(s![subslice.., ..]).to_owned();
    let slice: Vec<usize> = y.iter().map(|&x| x as usize).collect();
    let y_test = one_hot(&slice, 10).slice(s![subslice.., ..]).to_owned();
    let y_hat = sut.forward(&X_test);
    let accuracy = accuracy(&y_hat, &y_test);
    dbg!(X_test.raw_dim(), y_hat.dim());
    Some(accuracy)
}




#[cfg(test)]
mod tests {
    use crate::util::mnist_helper::{load_mnist, train_mnist};

    #[ignore]
    #[test]
    pub fn mnist() {
        let (X, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
        let accuracy = train_mnist(X,y).unwrap();
        assert!(accuracy>0.85)
    }
}