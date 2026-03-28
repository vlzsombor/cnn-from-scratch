use ndarray::{array, s, Array1, Array2, ArrayView2, Axis};
use crate::train::layer::Layer;
use crate::train::layerable::Layerable;
use crate::util::mnist_helper::{csv_to_image, csv_to_image_oned};

#[derive(Debug)]
pub struct ConvolutionalLayer {
    image_dimension: usize,
    layer: Layer
}
impl ConvolutionalLayer
{
    const input_dimension: usize = 32;
    pub fn new(layer: Layer, image_dimension: usize) -> Self
    {
        ConvolutionalLayer {
            image_dimension,
            layer
        }
    }
}
impl Layerable for ConvolutionalLayer
{
    fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let kernel: Array2<f32> = array![
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
];
//        let kernel = kernel.t();
        let first = x.row(0); //.unwrap();
        let twoD: Array2<f32> = first
            .into_shape_with_order((self.image_dimension, self.image_dimension))
            .unwrap()
            .to_owned();
        let twoD = pad(&twoD, 2, 0.0);
        dbg!(twoD.dim());
        csv_to_image(&twoD, "before.png");
        let x = twoD;

        let kernel_view: ArrayView2<f32> = kernel.view();
        let k_rows = kernel.nrows();
        let k_cols = kernel.ncols();

        let mut sub_res = Array2::zeros((x.nrows() - k_rows + 1, x.ncols() - k_cols + 1));

        for row_idx in 0..sub_res.nrows()
        {
            for col_idx in 0..sub_res.ncols()
            {
                let sub_x: ArrayView2<f32> =
                    x.slice(s!
                    [row_idx..row_idx+k_rows,
                        col_idx..col_idx+k_cols]
                    );
                let res = &sub_x * &kernel_view;
                sub_res[[row_idx, col_idx]] = res.sum();
            }
        }

        csv_to_image(&sub_res, "after.png");
        sub_res
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
