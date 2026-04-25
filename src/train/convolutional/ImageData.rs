use ndarray::{s, Array3, ArrayView2};
use serde::{Deserialize, Serialize};
use crate::train::convolutional::Kernel::Kernel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub image: Array3<f32>
}
impl ImageData{
    pub fn new(image: Array3<f32>) -> Self{
        ImageData{
            image
        }
    }
    pub fn get_channel_number(&self) -> usize {
        self.image.shape()[0]
    }
    pub fn get_row(&self) -> usize {
        self.image.shape()[1]
    }
    pub fn get_col(&self) -> usize {
        self.image.shape()[2]
    }
    pub fn get_image(&self, channel_number: usize) -> ArrayView2<f32>{
        self.image.slice(s![channel_number, .., ..])
    }

    pub fn rot180(&self) -> Self {
        Self::new(self.image.slice(s![.., ..;-1, ..;-1]).to_owned())
    }

    pub fn get_new_temp_array(&self, kernel: &Kernel) -> Array3<f32> {
        let (new_r, new_c) = self.get_image_size_after_convolution(kernel);
        Array3::zeros([kernel.get_output_channel_number(), new_r, new_c])
    }

    pub fn get_image_size_after_convolution_image_data(&self, image_data: &ImageData) -> (usize, usize) {
        (self.get_row() - image_data.get_row() + 1, self.get_col() - image_data.get_col() + 1)
    }
    pub fn get_image_size_after_convolution(&self, kernel: &Kernel) -> (usize, usize) {
        (self.get_row() - kernel.get_row() + 1, self.get_col() - kernel.get_col() + 1)
    }
}
