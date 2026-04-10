use std::fmt::{Debug, Formatter};
use std::ops::Range;
use image::codecs::png::CompressionType::Default;
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView2, ArrayView3, Axis, Ix3, OwnedRepr};
use crate::train::activation::ReluActivation;
use crate::train::layerable::Layerable;
use pipe::pipe;
const ALPHA: f32 = 0.0001;
#[derive(Debug)]
pub(crate) struct ConvolutionalMatlab
{
    kernel: Array4<f32>,
    biases: Array1<f32>,
    alpha: f32,
    cached_input: Option<Array3<f32>>
}

impl ConvolutionalMatlab
{
    pub fn new(kernel: Array4<f32>, alpha: f32) -> Self
    {
        let biases = Array1::zeros(kernel.shape()[0]);// bug in channels should be out channels
        ConvolutionalMatlab
        {
            kernel,
            biases,
            alpha,
            cached_input: None
        }
    }
    pub(crate) fn k1_sigma(&mut self, x: &Array3<f32>) -> Array3<f32>
    {
        self.k1(x).get_sigmoid()
    }
    /// k1 {1,6,5x5}
    pub(crate) fn k1(&mut self, x: &Array3<f32>) -> Array3<f32>
    {
        assert_eq!(x.shape()[0], self.kernel.shape()[0]);
        let mut return_res = Array3::zeros([self.kernel.shape()[1], x.shape()[1]-self.kernel.shape()[2] +1, x.shape()[2] -self.kernel.shape()[3] + 1]);
        for output_index in 0..self.kernel.shape()[1]{
            let mut acc = Array2::zeros((return_res.shape()[1], return_res.shape()[2]));
            for input_index in 0..self.kernel.shape()[0] {
                let sub_x = x.slice(s![input_index, .., ..]);
                let sub_k = self.kernel.slice(s![input_index, output_index, .., ..]);
                acc = acc + Self::conv2d(&sub_x, &sub_k);
            }
            return_res
                .slice_mut(s![output_index, ..,..])
                .assign(&acc);
        }
        let b = return_res + &self.biases;
        self.cached_input = Some(x.clone());
        b
    }

    pub fn map_conv_boundary(kernel_length: i32) -> (i32, i32) {
        let upper = kernel_length / 2;
        (-upper,upper)
    }
    pub fn conv2d(data: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
        let new_row = data.shape()[0] - kernel.shape()[0] + 1;
        let new_col = data.shape()[1] - kernel.shape()[1] + 1;
        let mut return_data: Array2<f32> = Array2::zeros((new_row, new_col));
        let k_row = kernel.shape()[0];
        let k_col = kernel.shape()[1];
        for i in 0..new_row {
            for j in 0..new_col {
                let sub_data = &data.slice(s![i..i+k_row, j..j+k_col]);
                let multiply = sub_data * kernel;
                return_data[[i,j]] = multiply.sum()
            }
        }
        return_data
    }
    pub fn conv(x: &Array3<f32>, kernel: &Array4<f32>) -> ArrayBase<OwnedRepr<f32>, Ix3> {
        let mut c1 = Array3::zeros((
            kernel.shape()[1],
            x.shape()[1] - kernel.shape()[2] + 1,
            x.shape()[2] - kernel.shape()[3] + 1
        )
        );
        for kernel_idx in 0..kernel.shape()[0] {
            for row in 0..c1.shape()[1] {
                for col in 0..c1.shape()[2] {
                    let value = x.slice(
                        s![kernel_idx,
                            row..row+kernel.shape()[2],
                            col..col+kernel.shape()[3]]
                    );
                    let sum = (&value * kernel).sum();
                    c1[[kernel_idx, row, col]] = sum;
                }
            }
        }
        c1
    }
    pub fn k1_back(&mut self, delta_c_x: &Array3<f32>) -> ArrayBase<OwnedRepr<f32>, Ix3> {
        let input = self.cached_input.as_ref().unwrap();
        let mut rot180 = Self::rot180(input);
        let delta_k = Array3::zeros((rot180.shape()[0], rot180.shape()[1], rot180.shape()[2]));
        for kernel in 0..rot180.shape()[0] {
            let rot = rot180.slice(s![kernel, ..,..]);
            let delta_c = delta_c_x.slice(s![kernel, ..,..]);
            let t = Self::conv2d(&rot, &delta_c);
            rot180
                .slice_mut(s![kernel,..,..])
                .assign(&t);
        }
        self.kernel = &self.kernel - self.alpha * &delta_k;
        delta_k
    }
    fn rot180(kernel: &Array3<f32>) -> Array3<f32> {
        kernel.slice(s![.., ..;-1, ..;-1]).to_owned()
    }    // fn convolution(x: &Array3<f32>, kernel: &Array4<f32>, biases: &Array1<f32>) -> Array3<f32>

    // {
    //     let k_rows = kernel.shape()[2];
    //     let k_cols = kernel.shape()[3];
    //     let mut c1_result = Array3::zeros((
    //         kernel.shape()[0],
    //         x.shape()[1] - k_rows + 1,
    //         x.shape()[2] - k_cols + 1)
    //     );
    //     for kernel_idx in 0..c1_result.shape()[0]{
    //         for row_idx in 0..c1_result.shape()[1]
    //         {
    //             for col_idx in 0..c1_result.shape()[2]
    //             {
    //                 let sub_x: ArrayView2<f32> =
    //                     x.slice(s!
    //                     [
    //                         kernel_idx,
    //                         row_idx..row_idx+k_rows,
    //                         col_idx..col_idx+k_cols
    //                     ]
    //                     );
    //                 let res = &sub_x * &kernel.slice(s![kernel_idx, .., .., ..]);
    //                 c1_result[[kernel_idx, row_idx, col_idx]] = res.sum() + biases[kernel_idx];
    //             }
    //         }
    //     }
    //     let c1_result = c1_result.get_relu();
    //     c1_result
    // }
}

// impl Layerable for ConvolutionalMatlab
// {
//     fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
//         let res = self.k1(x);
//
//         res
//     }



//     fn backward_propagation(&mut self, dc_da: &Array2<f32>) -> Array2<f32> {
//         todo!()
//     }
// }


#[cfg(test)]
mod tests {
    use crate::train::loss_functions::{cross_entropy_loss};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayView2};
    use crate::train::activation::ReluActivation;
    use crate::train::convolutional_matlab::{ConvolutionalMatlab, ALPHA};
    use crate::util::mnist_helper::load_mnist;
    #[test]
    pub fn c1_back_prop_test()
    {

        let kernel: Array4<f32> = array![
            [[
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
            ]]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28,28))
            .unwrap()
            .to_owned();

        let res = sut.k1(&a);
        let y :Array3<f32> = Array3::ones((1,24,24)) * 250.;
        for _ in 0..100{
            let y_hat = sut.k1(&a);
            let delta_y_hat = (&y_hat - &y) * &y_hat.get_sigmoid_derivative();
//            let res = sut.k1_back(&delta_y_hat);

 //           let dummy_loss = res.mapv(|x| x.powi(2)).sum();
        }


    }
    #[test]
    pub fn c1_size_test2()
    {

        let kernel: Array4<f32> = array![
            [[
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
            ]]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28,28))
            .unwrap()
            .to_owned();

        let res = sut.k1(&a);

        let aslice: &ArrayView2<f32> = &a.slice(s![0, 1..25,1..25]);
        let ressss: &ArrayView2<f32> = &res.slice(s![0,..,..]);

        let final_res = &a.slice(s![0, 16, 2..26]);
        let final_res2 = &ressss.row(14);
        assert_eq!(&final_res, &final_res2);
        //    assert_eq!(&aslice.shape(), &ressss.shape());
        assert_eq!(&aslice.row(0), &ressss.row(0));
    }
    #[test]
    pub fn c1_size_test()
    {
        let kernel: Array4<f32> = Array4::ones((1,6,5,5));
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28,28))
            .unwrap()
            .to_owned();
        let res = sut.k1(&a);
        assert_eq!(res.shape(), &[6, 24, 24]);
    }

    #[test]
    pub fn conv2d_test(){
        let kernel: Array2<f32> = array![
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array2<f32> = first.into_shape_with_order((28,28))
            .unwrap()
            .to_owned();

        let res = ConvolutionalMatlab::conv2d(&a.view(),&kernel.view());

        let aslice: &ArrayView2<f32> = &a.slice(s![1..25,1..25]);
        let ressss: &ArrayView2<f32> = &res.slice(s![..,..]);

        let final_res = &a.slice(s![16, 2..26]);
        let final_res2 = &ressss.row(14);
        assert_eq!(&final_res, &final_res2);
        //    assert_eq!(&aslice.shape(), &ressss.shape());
        assert_eq!(&aslice.row(0), &ressss.row(0));
    }
}
