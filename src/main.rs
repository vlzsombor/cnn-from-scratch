use ndarray::{array, Array2, Array3};
use crate::train::convolutional::CnnContainer::CnnContainer;
use crate::train::convolutional::ImageData::ImageData;
use crate::train::layer_container::{cross_entropy_loss_derivative_and_softmax, loss, mse_loss_derivative};
use crate::util::mnist_helper::load_mnist;

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
    let first = x.row(0); //.unwrap();
    let first = first.mapv(|x| x/255.);
    let first = first.view();
    let two_dimensional_data: Array2<f32> = first.into_shape_with_order((28, 28))
        .unwrap()
        .to_owned();

    let image_data: Array3<f32> = first.to_shape((1, 28, 28)).unwrap().to_owned();
    dbg!(&image_data.shape());
    let image_data_wrapper = ImageData::new(image_data);

    let target: Array2<f32> = array![
            [0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,]
        ];

    let mut y_hat = sut.forward(&image_data_wrapper);
    for _ in 0..1000{
        y_hat = sut.forward(&image_data_wrapper);
        let loss_derivative = mse_loss_derivative(&y_hat, &target);
//        let loss_derivative = cross_entropy_loss_derivative_and_softmax(&y_hat, &y);
        let loss = loss(&y_hat, &target);
        dbg!(&loss);
        sut.backward_linear(loss_derivative);
    }
    let a = sut.backward_linear(target);

}