//todo: 100 samples to 1 accuracy bring, get real loss function of cross entropy loss etc
extern crate core;
use crate::train::convolutional::CnnContainer::CnnContainer;
use crate::train::convolutional::ImageData::ImageData;
use crate::train::layer_container::{
    cross_entropy_loss, cross_entropy_loss_2, cross_entropy_loss_derivative_and_softmax,
};
use crate::util::mnist_helper::{load_mnist, load_mnist_without_label};
use crate::util::util::{accuracy, one_hot};
use image::GrayImage;
use ndarray::{Array2, Array3, ArrayView2, Axis, array, s};
use serde::Deserialize;
use std::fs::File;
use std::fs::{self};
use std::io;
use std::io::Write;
use std::path::PathBuf;
pub mod train;
pub mod util;

#[derive(Debug, Deserialize)]
struct CnnConfig {
    n_samples: i32,
    epochs: i32,
    writeOutEpoch: i32,
    learningRate: f32,
}

fn env_path() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "app/data/cnn_from_scratch/"
    }
}

// const FILE_PATH: &str = "";

fn get_file_path(file_path: &str) -> String {
    let r = format!("{}{}", env_path(), file_path);
    r
}
fn without_label(file_path: &str) {
    let train_size = 28_000;
    let config_size = 0; //cnn_config.n_samples;
    println!("{}", &train_size);

    let test_path = r#"C:\Users\ZsomborVeres-Lakos\Downloads\kaggle\test.csv\test.csv"#;
    // let mut sut = CnnContainer::new_default(config.learningRate);
    let x = load_mnist_without_label(get_file_path(test_path).as_str()).unwrap();
    let subset: Array2<f32> = x
        .slice(s![config_size..config_size + train_size, ..])
        .to_owned();
    let subset = subset.mapv(|x| x / 255.);
    let batch: Array3<f32> = subset
        .into_shape_with_order((train_size as usize, 28, 28))
        .unwrap()
        .to_owned();

    let json = std::fs::read_to_string(get_file_path(file_path).as_str()).unwrap();
    let mut sut: CnnContainer = serde_json::from_str(&json).unwrap();

    let mut v: Vec<usize> = Vec::new();
    for i in 0..train_size {
        let x = batch.slice(s![i..i + 1, .., ..]).to_owned();
        let image_data = ImageData::new(x);
        let one_hot = sut.forward(&image_data);
        let one_hot = one_hot.slice(s![0, ..]);
        let y_hat = one_hot
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        v.push(y_hat);
    }
    let mut file: File = File::create("submission.csv").unwrap();
    writeln!(file, "ImageId,Label").unwrap();
    for (i, label) in v.iter().enumerate() {
        writeln!(file, "{},{}", i + 1, label).unwrap();
    }
}

fn accuracy_mode(cnn_config: &CnnConfig, file_path: &str) {
    let train_size = 60_000;
    let config_size = 0; //cnn_config.n_samples;
    println!("{}", &train_size);

    // let mut sut = CnnContainer::new_default(config.learningRate);
    let test_path = r#"C:\Users\ZsomborVeres-Lakos\Downloads\kaggle\test.csv\test.csv"#;
    let (x, y) = load_mnist(get_file_path(test_path).as_str()).unwrap();
    let subset: Array2<f32> = x
        .slice(s![config_size..config_size + train_size, ..])
        .to_owned();
    let subset = subset.mapv(|x| x / 255.);
    let batch: Array3<f32> = subset
        .into_shape_with_order((train_size as usize, 28, 28))
        .unwrap()
        .to_owned();

    let json = std::fs::read_to_string(get_file_path(file_path).as_str()).unwrap();
    let mut sut: CnnContainer = serde_json::from_str(&json).unwrap();
    let slice: Vec<usize> = y
        .iter()
        .skip(config_size as usize)
        .take(train_size as usize)
        .map(|&x| x as usize)
        .collect();
    let target = one_hot(&slice, 10);
    let mut acc = 0.0;
    let mut lossv = 0.0;

    for i in 0..train_size {
        let x = batch.slice(s![i..i + 1, .., ..]).to_owned();
        let image_data = ImageData::new(x);
        let y_hat = sut.forward(&image_data);
        let target_f = target.slice(s![i..i + 1, ..]).to_owned();
        let loss_derivative = cross_entropy_loss_derivative_and_softmax(&y_hat, &target_f);

        // sut.backward_linear(loss_derivative.clone());
        acc += accuracy(&y_hat, &target_f);
        lossv += cross_entropy_loss_2(&y_hat, &target_f);

        if i % 1000 == 0 {
            println!("{}", i);
            println!("{} {}", acc, acc / i as f32);
        }
    }
    dbg!(&(acc / (train_size) as f32));
    dbg!(&(lossv / (train_size) as f32));
}

fn main() {
    let settings = config::Config::builder()
        .add_source(config::File::with_name(&get_file_path("Settings")))
        .build()
        .unwrap();
    let config: CnnConfig = settings.try_deserialize().unwrap();

    let file_path = "results/model-69-5801.json";
    without_label(file_path);
    return;
    accuracy_mode(&config, file_path);
    println!("{:#?}", config);
    println!("hello2:wq");
    let n_samples = config.n_samples;
    let epochs = config.epochs; // 16000;
    let writeout_epoch = config.writeOutEpoch;
    println!("n_samples: {} epochs: {}", n_samples, epochs);

    let mut sut = CnnContainer::new_default(config.learningRate);
    let (x, y) = load_mnist(get_file_path("src/data/mnist_train_small.csv").as_str()).unwrap();
    let subset: Array2<f32> = x.slice(s![..n_samples, ..]).to_owned();
    let subset = subset.mapv(|x| x / 255.);
    let batch: Array3<f32> = subset
        .into_shape_with_order((n_samples as usize, 28, 28))
        .unwrap()
        .to_owned();

    let json = std::fs::read_to_string(get_file_path("model.json").as_str()).unwrap();
    let mut sut: CnnContainer = serde_json::from_str(&json).unwrap();
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
        if epoch % writeout_epoch == 0 || epoch == epochs {
            dbg!(&(acc / (n_samples * writeout_epoch) as f32));
            dbg!(&(lossv / (n_samples * writeout_epoch) as f32));
            println!("{} ====================", epoch);
            let json = serde_json::to_string(&sut).unwrap();
            let name = format!("results/model-{}-{}.json", epoch, acc);
            std::fs::write(get_file_path(&name), &json).unwrap();
            acc = 0.;
            lossv = 0.;
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
