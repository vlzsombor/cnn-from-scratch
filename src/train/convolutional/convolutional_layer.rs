use crate::train::activation::ReluActivation;
use crate::train::layer::{xavier, Activation, Layer};
use crate::train::layerable::Layerable;
use crate::util::mnist_helper::csv_to_image;
use ndarray::{array, s, Array1, Array2, Array3, Array4, ArrayView2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
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
            biases[k] = *xavier(1, 1).first().unwrap();
        }

        ConvolutionalLayer {
            image_dimension,
            layer,
            kernels,
            biases
        }
    }
    fn sub_sampling(&mut self, x: &Array3<f32>, sub_sampling_kernel_size: usize) -> Array3<f32> {
        let nrow = x.shape()[1] / sub_sampling_kernel_size;
        let ncolumn = x.shape()[1] / sub_sampling_kernel_size;
        let mut sub_sampled: Array3<f32> = Array3::zeros((x.shape()[0], nrow, ncolumn));

        for kernel_idx in 0..sub_sampled.shape()[0] {
            for row in 0..sub_sampled.shape()[1] {
                for col in 0..sub_sampled.shape()[2] {
                    let r = &x.slice(s![kernel_idx,
                        row..row+sub_sampling_kernel_size,
                        col..col+sub_sampling_kernel_size]
                    );
                    sub_sampled[[kernel_idx, row, col]] = r.mean().unwrap();
                }
            }
        }
        sub_sampled.get_relu()
    }

    fn map_idx(idx: usize) -> usize {
        match idx {
            6.. => idx - 6,
            _ => idx
        }
    }
    pub fn kernel_idx_to_input_idx(idx: usize) -> Vec<usize> {
        match idx {
            0..6 => vec![Self::map_idx(idx), Self::map_idx(idx + 1), Self::map_idx(idx + 2)],
            6..12 => vec![Self::map_idx(idx), Self::map_idx(idx + 1 - 6), Self::map_idx(idx + 2 - 6), Self::map_idx(idx + 3 - 6)],
            12 => vec![0, 1, 3, 4],
            13 => vec![1, 2, 4, 5],
            14 => vec![0, 2, 3, 5],
            15 => vec![0, 1, 2, 3, 4, 5],
            _ => todo!(),
        }
    }

    fn c3(&self, s2: &Array3<f32>) -> Array3<f32>
    {
        //n_kernel, n_input, rows, cols
        let mut kernels: Array4<f32> = Array4::zeros((16, 6, 5, 5));
        let mut biases: Array1<f32> = Array1::zeros((kernels.shape()[0]));
        let kernel_size = kernels.shape()[2];
        for k in 0..kernels.shape()[0] {
            kernels
                .slice_mut(s![k, 0, .., ..])
                .assign(&xavier(
                    kernel_size,
                    kernel_size)
                );
            biases[k] = *xavier(1, 1).first().unwrap();
        }
        let mut c3: Array3<f32> = Array3::zeros(
            (16, s2.shape()[1] - kernels.shape()[2] + 1, s2.shape()[2] - kernels.shape()[3] + 1)
        );
        for kernel_idx in 0..c3.shape()[0] {
            for row_idx in 0..c3.shape()[1]
            {
                for col_idx in 0..c3.shape()[2]
                {
                    let ress: f32 = Self::kernel_idx_to_input_idx(kernel_idx)
                        .iter()
                        .map(|&x| {
                            let patch = s2.slice(s![x, row_idx..row_idx+kernel_size, col_idx..col_idx+kernel_size]);
                            let kernel = kernels.slice(s![kernel_idx, x, .., ..]);
                            (&patch * &kernel).sum()
                        })
                        .sum();
                    c3[[kernel_idx, row_idx, col_idx]] = Activation::Relu_scalar(ress + biases[kernel_idx]);
                }
            }
        }
        c3
    }

    fn c5(&self, s2: &Array3<f32>) -> Array3<f32>
    {
        //n_kernel, n_input, rows, cols
        let mut kernels: Array4<f32> = Array4::zeros((120, 16, 5, 5));
        let mut biases: Array1<f32> = Array1::zeros((kernels.shape()[0]));
        let kernel_size = kernels.shape()[2];
        for k in 0..kernels.shape()[0] {
            kernels
                .slice_mut(s![k, 0, .., ..])
                .assign(&xavier(
                    kernel_size,
                    kernel_size)
                );
            biases[k] = *xavier(1, 1).first().unwrap();
        }
        let mut c3: Array3<f32> = Array3::zeros(
            (kernels.shape()[0], s2.shape()[1] - kernels.shape()[2] + 1, s2.shape()[2] - kernels.shape()[3] + 1)
        );
        for kernel_idx in 0..c3.shape()[0] {
            for row_idx in 0..c3.shape()[1]
            {
                for col_idx in 0..c3.shape()[2]
                {
                    let ress: f32 = (0..16)
                        .map(|x| {
                            let patch = s2.slice(s![x, row_idx..row_idx+kernel_size, col_idx..col_idx+kernel_size]);
                            let kernel = kernels.slice(s![kernel_idx, x, .., ..]);
                            (&patch * &kernel).sum()
                        })
                        .sum();
                    c3[[kernel_idx, row_idx, col_idx]] = Activation::Relu_scalar(ress + biases[kernel_idx]);
                }
            }
        }
        c3
    }
}

#[typetag::serde]
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
        let c1_result = &c1_result.get_relu();
/////////////////////////c1
        let s2 = self.sub_sampling(&c1_result, 2);
////////////////////////s2 ↑
        let c3 = self.c3(&s2);
        let s4 = self.sub_sampling(&c3, 2);
        let c5 = self.c5(&s4);
        if(!is_effectively_1d(&c5)){
            todo!()
        }
        let flat_view = c5.to_shape((120,)).unwrap();

        let twodreturn: Array2<f32> = flat_view.to_shape((1,120)).unwrap().into_owned();
        twodreturn
    }
    fn backward_propagation(&mut self, dc_da: &Array2<f32>) -> Array2<f32> {
        todo!()
    }
}
fn is_effectively_1d(arr: &Array3<f32>) -> bool {
    arr.shape()[1] == 1 && arr.shape()[2] == 1
}

fn pad(input: &Array2<f32>, pad: usize, fill: f32) -> Array2<f32> {
    let (rows, cols) = input.dim();
    let mut output = Array2::from_elem((rows + 2 * pad, cols + 2 * pad), fill);
    output
        .slice_mut(s![pad..pad + rows, pad..pad + cols])
        .assign(input);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_idx() {

        let a = (0..16)
            .for_each(|x|{
                let mut r = ConvolutionalLayer::kernel_idx_to_input_idx(x);
                let mut b = kernel_idx_to_input_idx(x);
                r.sort();
                b.sort();
                assert_eq!(r, b)
            });
    }

    fn kernel_idx_to_input_idx(idx: usize) -> Vec<usize> {
        match idx {
            0  => vec![0, 1, 2],
            1  => vec![1, 2, 3],
            2  => vec![2, 3, 4],
            3  => vec![3, 4, 5],
            4  => vec![4, 5, 0],
            5  => vec![5, 0, 1],
            6  => vec![0, 1, 2, 3],
            7  => vec![1, 2, 3, 4],
            8  => vec![2, 3, 4, 5],
            9  => vec![ 3, 4, 5, 0],
            10 => vec![0, 1, 4, 5],
            11 => vec![0, 1, 2, 5],
            12 => vec![0, 1, 3, 4],
            13 => vec![1, 2, 4, 5],
            14 => vec![0, 2, 3, 5],
            15 => vec![0, 1, 2, 3, 4, 5],
            _  => panic!("invalid kernel index"),
        }
    }

    //    #[test]
    //    fn test_layer_stats_normal() {
    //        let layer = Layer::new(100, 50);
    //        let mean = layer.weights.mean().unwrap();
    //        let std = layer.weights.std(0.); // population std
    //
    //        // Rough checks for StandardNormal (μ=0, σ=1)
    //        assert!((-0.2..=0.2).contains(&mean));
    //        assert!((0.8..=1.2).contains(&std));
    //    }
}
