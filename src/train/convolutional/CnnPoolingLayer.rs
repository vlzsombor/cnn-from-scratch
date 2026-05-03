use ndarray::{s, Array1, Array3, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::ImageData::ImageData;

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct CnnPoolingLayer{}

impl CnnPoolingLayer {
    pub fn new() -> Self{
        CnnPoolingLayer{}
    }
}

#[typetag::serde]
impl CnnLayerable for CnnPoolingLayer{
    fn forward_propagation(&mut self, x: &ImageData) -> Array3<f32> {
        let mut pooled = Array3::zeros((x.get_channel_number(),x.get_row()/2,x.get_col()/2));
        for row in 0..pooled.shape()[1]{
            for col in 0..pooled.shape()[2]{
                let window = x.image.slice(s![.., row*2..row*2+2, col*2..col*2+2]);
                let mean: Array1<f32> = window
                    .mean_axis(Axis(1)).unwrap()
                    .mean_axis(Axis(1)).unwrap();
                pooled.slice_mut(s![.., row, col]).assign(&mean);
            }
        }
        pooled
    }

    fn backward_propagation(&mut self, delta_c: &ImageData) -> Array3<f32> {
        let (c, h, w) = (
            delta_c.get_channel_number(),
            delta_c.get_row(),
            delta_c.get_col(),
        );
        let mut de_pooled: Array3<f32> = Array3::zeros((c, h * 2, w * 2));
        for row in 0..h * 2 {
            for col in 0..w * 2 {
                let src: ArrayView1<f32> = delta_c.image.slice(s![.., row / 2, col / 2]);
                de_pooled.slice_mut(s![.., row, col]).assign(&(&src * 0.25_f32));
            }
        }
        de_pooled
    }


}
