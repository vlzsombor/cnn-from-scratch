use crate::train::layer::{ActivationLayer, Layer};
use crate::train::layer_container::{cross_entropy_loss_and_softmax, LayerContainer};
use crate::train::loss_functions::softmax;
use crate::util::util::{accuracy, one_hot};
use ndarray::{s, Array1, Array2};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::train::convolutional::convolutional_layer;
use crate::train::convolutional::convolutional_layer::ConvolutionalLayer;

pub fn load_mnist(path: &str) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
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



pub fn train_mnist_cnn(x: Array2<f32>, y: Array1<f32>) -> Option<f32>
{
    dbg!(x.raw_dim());
    let subslice = 1;
    let x_train = normalize_mnist(&x).slice(s![..subslice, ..]).to_owned();
    let slice: Vec<usize> = y.iter().map(|&x| x as usize).collect();
    let y_train = one_hot(&slice, 10).slice(s![..subslice, ..]).to_owned();

    let layer = Layer::new(28*28, 128, 0.0001);
    let aaa = ConvolutionalLayer::new(layer, 28);
    let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
        Box::new(aaa),
        Box::new(ActivationLayer::relu()),
    ];
    let mut sut = LayerContainer::new_layers_boxed(layers);
    for i in 0..500{
        let y_hat = sut.forward(&x_train);
        sut.backward_propagation(cross_entropy_loss_and_softmax(&y_hat, &y_train));
        if i % 100 == 0  {
            let accuracy = accuracy(&y_hat, &y_train);
            dbg!(i, accuracy);
        }
    }
    let y_hat = sut.forward(&x_train);

    dbg!(softmax(y_hat.view()).row(0));
    let X_test = normalize_mnist(&x).slice(s![subslice.., ..]).to_owned();
    let slice: Vec<usize> = y.iter().map(|&x| x as usize).collect();
    let y_test = one_hot(&slice, 10).slice(s![subslice.., ..]).to_owned();
    let y_hat = sut.forward(&X_test);
    let accuracy = accuracy(&y_hat, &y_test);
    dbg!(X_test.raw_dim(), y_hat.dim());
    Some(accuracy)
}
pub fn train_mnist(x: Array2<f32>, y: Array1<f32>) -> Option<f32>
{
    let subslice = 1_000;
    let X_train = normalize_mnist(&x).slice(s![..subslice, ..]).to_owned();
    let slice: Vec<usize> = y.iter().map(|&x| x as usize).collect();
    let y_train = one_hot(&slice, 10).slice(s![..subslice, ..]).to_owned();
    let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
        Box::new(Layer::new(28*28, 128, 0.0001)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(128, 128,  0.0001)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(128, 10, 0.0001)),
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
    let X_test = normalize_mnist(&x).slice(s![subslice.., ..]).to_owned();
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
        dbg!(accuracy);
        assert!(accuracy>0.85)
    }
}


use image::{GrayImage, Luma};

pub fn csv_to_image_oned(pixels: &Array1<f32>, path: &str) {
    let twod = &pixels.clone().into_shape_with_order((28,28)).unwrap();
    csv_to_image(twod, path)
}

pub fn csv_to_image(pixels: &Array2<f32>, path: &str) {
    let (rows, cols) = pixels.dim();
    let mut img = GrayImage::new(cols as u32, rows as u32);

    for i in 0..rows {
        for j in 0..cols {
            // LeNet-5 Normalisierung rückgängig machen: [-0.1, 1.175] → [0, 255]
            let val = ((pixels[[i, j]] + 0.1) / 1.275 * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(j as u32, i as u32, Luma([val]));
        }
    }

    img.save(path).unwrap();
}
