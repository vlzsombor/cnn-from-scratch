use ndarray::{array, s, Array1, Array2, Array3, Array4, ArrayView2, Axis};
use crate::train::layer::{xavier, Layer};
use crate::train::layerable::Layerable;
use crate::util::mnist_helper::{csv_to_image, csv_to_image_oned};

#[derive(Debug)]
pub struct ConvolutionalLayer {
    image_dimension: usize,
    layer: Layer,
    kernels: Array4<f32>,
    biases: Array1<f32>
}
impl ConvolutionalLayer
{
    const input_dimension: usize = 32;
    pub fn new(layer: Layer, image_dimension: usize) -> Self
    {
        // C5 Kernel:  [n_filters, n_channels, height, width]
        let a = xavier(28usize, 28usize);
        let kernel_size = 5;
        let kernel_num = 6;
        let mut kernels: Array4<f32> = Array4::zeros((kernel_num, 1, kernel_size, kernel_size));
        let mut biases: Array1<f32> = Array1::zeros(kernel_num);
        for k in 0..kernels.shape()[0] {
            kernels
                .slice_mut(s![k, 0, .., ..])
                .assign(&xavier(
                    kernel_size as usize,
                    kernel_size as usize)
                );
            biases[k] = *xavier(1,1).first().unwrap();
        }

        ConvolutionalLayer {
            image_dimension,
            layer,
            kernels,
            biases
        }
    }
    fn sub_sampling(&mut self, x: &Array3<f32>, sub_sampling_kernel_size: usize) -> Array3<f32> {
        dbg!(&x.shape());
        let nrow = x.shape()[1] / sub_sampling_kernel_size;
        let ncolumn = x.shape()[1] / sub_sampling_kernel_size;
        let mut sub_sampled: Array3<f32> = Array3::zeros((x.shape()[0], nrow, ncolumn));

        for kernel_idx in 0..sub_sampled.shape()[0] {
            for row in 0..sub_sampled.shape()[1] {
                for col in 0..sub_sampled.shape()[2]{
                    let r = &x.slice(s![kernel_idx,
                        row..row+sub_sampling_kernel_size,
                        col..col+sub_sampling_kernel_size]
                    );
                    dbg!(r.shape());
                    sub_sampled[[kernel_idx, row, col]] = r.mean().unwrap();
                }
            }
        }
        sub_sampled
    }
}
impl Layerable for ConvolutionalLayer
{

    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {

        let kernelxd: Array2<f32> = array![
            [0., -1., 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 0.0]
        ];

        let first = x.row(0); //.unwrap();
        let twoD: Array2<f32> = first
            .into_shape_with_order((self.image_dimension, self.image_dimension))
            .unwrap()
            .to_owned();
        let twoD = pad(&twoD, 2, 0.0);
        dbg!(twoD.dim());
        csv_to_image(&twoD, "before.png");
        let x = twoD;

//        let kernel_view: ArrayView2<f32> = kernel.view();
        let k_rows = self.kernels.shape()[2];
        let k_cols = self.kernels.shape()[3];
/////////////C1
        let mut c1_result = Array3::zeros((
            self.kernels.shape()[0],
            x.nrows() - k_rows + 1,
            x.ncols() - k_cols + 1)
        );
        dbg!(c1_result.shape());
        dbg!(x.nrows(), x.ncols(),  x.nrows() - k_rows + 1);
        dbg!(c1_result.shape());



        for kernel_idx in 0..c1_result.shape()[0]{
            for row_idx in 0..c1_result.shape()[1]
            {
                for col_idx in 0..c1_result.shape()[2]
                {
                    let sub_x: ArrayView2<f32> =
                        x.slice(s!
                        [row_idx..row_idx+k_rows,
                        col_idx..col_idx+k_cols]
                        );
                    let res = &sub_x * &self.kernels.slice(s![kernel_idx,0, ..,..]);
                    c1_result[[kernel_idx, row_idx, col_idx]] = res.sum() + self.biases[kernel_idx];
                }
            }
        }
/////////////////////////c1
        let sub = self.sub_sampling(&c1_result, 2);


        dbg!(sub.shape());


//        csv_to_image(&sub_res, "after.png");
//        sub_res
        todo!()
    }


    fn backward_propagation(&mut self, dc_da: &Array2<f32>) -> Array2<f32> {
        todo!()
    }
}
fn pad(input: &Array2<f32>, pad: usize, fill: f32) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let mut output = Array2::from_elem((rows + 2 * pad, cols + 2 * pad), fill);
    output
        .slice_mut(s![pad..pad + rows, pad..pad + cols])
        .assign(input);
    output
}
