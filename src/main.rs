//todo: 100 samples to 1 accuracy bring, get real loss function of cross entropy loss etc
extern crate core;

use crate::train::convolutional::CnnContainer::CnnContainer;
use crate::train::convolutional::ImageData::ImageData;
use crate::train::layer_container::{
    cross_entropy_loss, cross_entropy_loss_2, cross_entropy_loss_derivative_and_softmax,
};
use crate::util::mnist_helper::load_mnist;
use crate::util::util::{accuracy, one_hot};
use image::GrayImage;
use ndarray::{Array2, Array3, ArrayView2, Axis, array, s};

pub mod train;
pub mod util;

fn main() {
    let mut sut = CnnContainer::new_default();
    let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
    let n_samples: i32 = 160;
    let epochs: i32 = 100; // 16000;
    let subset: Array2<f32> = x.slice(s![25..25 + n_samples, ..]).to_owned();
    let subset = subset.mapv(|x| x / 255.);
    let batch: Array3<f32> = subset
        .into_shape_with_order((n_samples as usize, 28, 28))
        .unwrap()
        .to_owned();

    let json = std::fs::read_to_string("model.json").unwrap();
    let mut sut: CnnContainer = serde_json::from_str(&json).unwrap();
    println!("hello");
    let slice: Vec<usize> = y
        .iter()
        .take(n_samples as usize)
        .map(|&x| x as usize)
        .collect();
    let target = one_hot(&slice, 10);
    let mut acc = 0.0;
    let mut lossv = 0.0;
    for epoch in 1..=epochs {
        for i in 0..n_samples {
            let x = batch.slice(s![i..i + 1, .., ..]).to_owned();
            let image_data = ImageData::new(x);
            let y_hat = sut.forward(&image_data);
            let target_f = target.slice(s![i..i + 1, ..]).to_owned();
            let loss_derivative = cross_entropy_loss_derivative_and_softmax(&y_hat, &target_f);

            sut.backward_linear(loss_derivative.clone());
            acc += accuracy(&y_hat, &target_f);
            lossv += cross_entropy_loss_2(&y_hat, &target_f);
        }
        let writeout_epoch = 10;
        if epoch % writeout_epoch == 0 || epoch == epochs {
            dbg!(&(acc / (n_samples * writeout_epoch) as f32));
            dbg!(&(lossv / (n_samples * writeout_epoch) as f32));
            println!("{} ====================", epoch);
            let json = serde_json::to_string(&sut).unwrap();
            let name = format!("model-{}-{}.json", epoch, acc);
            std::fs::write(name, &json).unwrap();
            acc = 0.;
            lossv = 0.;
            let json = serde_json::to_string(&sut).unwrap();
            std::fs::write("model.json", &json).unwrap();
        }
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
