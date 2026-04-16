extern crate core;

use ndarray::{array, s, Array3, Array4, ArrayView2};
use crate::train::convolutional_layer::ConvolutionalLayer;
use crate::train::convolutional_matlab::{ConvolutionalMatlab, ImageData};
use crate::train::layer::{ActivationLayer, Layer};
use crate::train::layer_container::LayerContainer;
use crate::util::mnist_helper::{load_mnist, train_mnist, train_mnist_cnn};

pub mod util;
pub mod train;


fn main() {
    //Test model: kernel 2,1,3x3
    let epsilon = 1e-3_f32;
    let alpha = 1.;
    let kernel: Array4<f32> = {
        let mut k = Array4::zeros((1,3,3,3));
        k.slice_mut(s![..,..,1,1]).fill(1.0);
        k
    };
    //1 5x5
    let input_image_raw: Array3<f32> = Array3::ones((1,5,5)) * 0.1;
    let input_image = ImageData::new(input_image_raw);

    let target_image_raw: Array3<f32> = Array3::ones((3,3,3)) * 0.5;
    let target_image = ImageData::new(target_image_raw);
    let mut sut = ConvolutionalMatlab::new(kernel.clone(), alpha);
    sut.k1(&input_image);
    let backward = sut.compute_kernel_gradient(&target_image);
    assert_eq!(backward.raw_dim(), kernel.raw_dim());
    let size= (&kernel.clone()).shape()[2];
    let in_size= (&kernel.clone()).shape()[0];
    let out_size= (&kernel.clone()).shape()[1];
    let mut i = 0;
    for inp in 0..in_size {
        for out in 0..out_size {
            for row in 0..size {
                for col in 0..size {
                    let kernel = kernel.clone();
                    let idx = (inp,out,row,col);
                    let mut sut_plus = ConvolutionalMatlab::new({
                                                                    let mut k = kernel.clone();
                                                                    k[idx] += epsilon;
                                                                    k.clone()
                                                                }, alpha);

                    let mut sut_minus = ConvolutionalMatlab::new({
                                                                     let mut k = (&kernel).clone();
                                                                     k[idx] -= epsilon;
                                                                     k
                                                                 }, alpha);

                    let f_plus = sut_plus.k1(&input_image);
                    let f_minus = sut_minus.k1(&input_image);
                    let loss_plus = ConvolutionalMatlab::compute_mse(
                        &f_plus, &target_image.image
                    );
                    let loss_minus = ConvolutionalMatlab::compute_mse(
                        &f_minus, &target_image.image
                    );
                    let numeric_grad = (loss_plus - loss_minus) / (2. * epsilon);

                    let mut sut = ConvolutionalMatlab::new(kernel, alpha);

                    let f = sut.k1(&input_image);

                    let delta = &f - &target_image.image;
                    let analytic = sut.compute_kernel_gradient(&ImageData::new(delta));
                    let analytic_grad= analytic[idx];
                    let relative_error = (&analytic_grad - &numeric_grad).abs() / &analytic_grad.abs().max(numeric_grad.abs());
                    assert!(relative_error < 1e-3,
                            "Gradient check failed: {analytic_grad} vs {numeric_grad} relative: {relative_error}");
                    dbg!(&relative_error, i);
                    i += 1;
                }
            }
        }
    }

}