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
    // let mut sut = CnnContainer::new_default();
    // let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
    // let first = x.row(0); //.unwrap();
    // let first = first.mapv(|x| x/255.);
    // let first = first.view();
    // let two_dimensional_data: Array2<f32> = first.into_shape_with_order((28, 28))
    //     .unwrap()
    //     .to_owned();
    //
    // let image_data: Array3<f32> = first.to_shape((1, 28, 28)).unwrap().to_owned();
    // dbg!(&image_data.shape());
    // let image_data_wrapper =ImageData::new(image_data);
    //
    // let target: Array2<f32> = array![
    //         [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,]
    //     ];
    //
    // let mut y_hat = sut.forward(&image_data_wrapper);
    // for _ in 0..100{
    //     y_hat = sut.forward(&image_data_wrapper);
    //     let a = mse_loss_derivative(&y_hat, &target);
    //     dbg!(&a);
    //     sut.backward_linear(a);
    // }
    // let a = sut.backward_linear(target);
    //
    // return;
    let mut sut = CnnContainer::new_default();
    let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
//    let first = x.row(0); //.unwrap();
    let n_samples = 10;
    let subset: Array2<f32> = x.slice(s![..n_samples, ..]).to_owned();
    let subset = subset.mapv(|x| x/255.);
 //   let first = first.mapv(|x| x/255.);
//    let first = first.view();
    let batch: Array3<f32> = subset.into_shape_with_order((n_samples, 28, 28))
        .unwrap()
        .to_owned();

    let slice: Vec<usize> = y.iter().take(n_samples).map(|&x| x as usize).collect();
    let target = one_hot(&slice, 10);
    for epoch in 0..1000{
        for i in 0..n_samples{
            let x = batch.slice(s![i..i+1, ..,..]).to_owned();
            let image_data = ImageData::new(x);
            let y_hat = sut.forward(&image_data);
            let target = target.slice(s![i..i+1,..]).to_owned();
//            let target = target.insert_axis(Axis(0)).to_owned();
            let loss_derivative = mse_loss_derivative(&y_hat, &target);
            //        let loss_derivative = cross_entropy_loss_derivative_and_softmax(&y_hat, &y);
            if epoch % 10 == 0 {
                let loss = loss(&y_hat, &target);
                dbg!(&loss);
            }
            sut.backward_linear(loss_derivative);
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
