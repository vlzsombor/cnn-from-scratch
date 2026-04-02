use std::fmt::{Debug, Formatter};
use std::ops::Range;
use ndarray::{array, s, Array, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3};
use crate::train::activation::ReluActivation;
use crate::train::layerable::Layerable;

#[derive(Debug)]
pub(crate) struct ConvolutionalMatlab
{
}

#[derive(Debug)]
struct ConvData{
    data: Array3<f32>
}
impl ConvData{
    pub fn new(data: Array3<f32>) -> Self{
        ConvData{
            data
        }
    }
}

impl ConvolutionalMatlab
{
    pub fn new() -> Self
    {
        ConvolutionalMatlab
        {
        }
    }
    /// k1 {1,6,5x5}
    pub(crate) fn k1(&self, x: &Array3<f32>, kernel: &Array4<f32>) -> Array3<f32>
    {
//        let k1: Array4<f32> = Array4::ones((1,6,5,5));

        // Array4::ones((1,6,5,5));
        let biases: Array1<f32> = Array1::zeros(kernel.shape()[1]);

        let mut c1 = Array3::zeros((
            kernel.shape()[1],
            x.shape()[1] - kernel.shape()[2] + 1,
            x.shape()[2] - kernel.shape()[3] + 1
        )
        );
//        dbg!(&x.slice(s![0, 14, ..]));
        let (b1, b2) = &Self::conv_border_count(&kernel, &c1);
        for kernel_idx in 0..kernel.shape()[0] {
            for row in 0..c1.shape()[1] {
                for col in 0..c1.shape()[2] {
                    let value = x.slice(
                        s![kernel_idx,
                            row..row+kernel.shape()[2],
                            col..col+kernel.shape()[3]]
                    );
                    let sum = (&value * kernel).sum();
                    if(row == 14 && col > 22){
                        // dbg!(&x.slice(s![0, row,col]));
                        // dbg!(&value, &sum, &row, &b1.start);
                    }
                    c1[[kernel_idx, row, col]] = sum + biases[kernel_idx];

                    // dbg!(&c1[[kernel_idx, row, col]]);
                }
//                dbg!(&row, &c1.slice(s![kernel_idx, row, ..]));
          //      dbg!()
            }
        }

        let sigma = c1;//.get_relu();
        sigma
    }
    fn conv_border_count(kernel: &Array4<f32>, c1: &Array3<f32>) -> (Range<usize>, Range<usize>) {
        let cborder = kernel.shape()[2]/2;
        let rborder = kernel.shape()[3]/2;
        (cborder..c1.shape()[1] - cborder, rborder..c1.shape()[2] - rborder)
    }
    // fn convolution(x: &Array3<f32>, kernel: &Array4<f32>, biases: &Array1<f32>) -> Array3<f32>

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
    use ndarray::{array, s, Array, Array1, Array3, Array4, ArrayView2};
    use crate::train::convolutional_matlab::ConvolutionalMatlab;
    use crate::util::mnist_helper::load_mnist;

    #[test]
    pub fn c1_size_test2()
    {

        let sut = crate::train::convolutional_matlab::ConvolutionalMatlab::new();
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28,28))
            .unwrap()
            .to_owned();

        let kernel: Array4<f32> = array![
            [[
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
            ]]
        ];
        let res = sut.k1(&a, &kernel);

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
        let sut = crate::train::convolutional_matlab::ConvolutionalMatlab::new();
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28,28))
            .unwrap()
            .to_owned();
        let kernel: Array4<f32> = Array4::ones((1,6,5,5));
        let res = sut.k1(&a, &kernel);
        assert_eq!(res.shape(), &[6, 24, 24]);
    }

}
