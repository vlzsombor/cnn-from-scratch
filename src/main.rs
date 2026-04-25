use std::io::Write;
use std::ops::Deref;
use image::GrayImage;
use ndarray::{array, s, Array2, Array3, ArrayView2, Axis};
use crate::train::convolutional::CnnContainer::CnnContainer;
use crate::train::convolutional::ImageData::ImageData;
use crate::train::layer_container::{cross_entropy_loss_derivative_and_softmax, loss, mse_loss_derivative};
use crate::util::mnist_helper::load_mnist;
use crate::util::util::one_hot;

pub mod util;
pub mod train;


fn main() {

    let mut sut = CnnContainer::new_default();
    let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
    let n_samples = 10;
    let subset: Array2<f32> = x.slice(s![..n_samples, ..]).to_owned();
    let subset = subset.mapv(|x| x/255.);
    let batch: Array3<f32> = subset.into_shape_with_order((n_samples, 28, 28))
        .unwrap()
        .to_owned();

    let slice: Vec<usize> = y.iter().take(n_samples).map(|&x| x as usize).collect();
    let target = one_hot(&slice, 10);
    for epoch in 0..=25{
        for i in 0..n_samples{
            let x = batch.slice(s![i..i+1, ..,..]).to_owned();
            let image_data = ImageData::new(x);
            let y_hat = sut.forward(&image_data);
            let target_f = target.slice(s![i..i+1,..]).to_owned();
            let loss_derivative = mse_loss_derivative(&y_hat, &target_f);
           sut.backward_linear(loss_derivative);
            if epoch % 25 == 0 {
                let y_hat_post = sut.forward(&image_data);
                let loss = loss(&y_hat_post, &target_f);
                dbg!(&loss);
            }
        }
        println!("{} ====================", epoch);
    }
    let bytes = bincode::serialize(&sut).unwrap();
    std::fs::write("model.bin", &bytes).unwrap();
    let bytes = std::fs::read("model.bin").unwrap();
    let mut back: CnnContainer = bincode::deserialize(&bytes).unwrap();
    println!("====================result");
    for epoch in 0..1{
        for i in 0..n_samples{
            let x = batch.slice(s![i..i+1, ..,..]).to_owned();
            let image_data = ImageData::new(x);
            let y_hat = back.forward(&image_data);
            let target_f = target.slice(s![i..i+1,..]).to_owned();
            let loss = loss(&y_hat, &target_f);
            dbg!(&loss);
        }
        println!("{} ====================", epoch);
    }
}


fn save_image(arr: &ArrayView2<f32>, path: &str) {
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let pixels: Vec<u8> = arr
        .iter()
        .map(|&x| (x * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(w as u32, h as u32, pixels)
        .unwrap()
        .save(path)
        .unwrap();
}
