use ndarray::{s, Array4, ArrayView2};
use crate::train::convolutional::ImageData::ImageData;

#[derive(Debug, Clone)]
pub struct Kernel{
    kernel: Array4<f32>
}

impl Kernel{

    pub fn subtract(&mut self, kernel2: &Array4<f32>){
        self.kernel = &self.kernel - kernel2;
    }
    pub fn get_new_temp_array(&self, imageData: &ImageData) -> Array4<f32> {
        let (new_r, new_c) = imageData.get_image_size_after_convolution(&self);
        dbg!(imageData.get_row(), imageData.get_col());
        dbg!(self.get_row(), self.get_col());
        Array4::zeros([self.get_input_channel_number(), self.get_output_channel_number(), new_r, new_c])
        //Array3::zeros([kernel.get_output_channel_number(), new_r, new_c])
    }
    pub fn get_shape(&self) -> &[usize] {
        self.kernel.shape()
    }
    pub fn get_row(&self) -> usize {
        self.kernel.shape()[2]
    }
    pub fn get_col(&self) -> usize {
        self.kernel.shape()[3]
    }
    pub fn get_image(&self, input: usize, output: usize) -> ArrayView2<f32>{
        self.kernel.slice(s![input, output, .., ..])
    }
    pub fn get_input_channel_number(&self) -> usize {
        self.kernel.shape()[0]
    }
    pub fn get_output_channel_number(&self) -> usize {
        self.kernel.shape()[1]
    }

    pub fn new(kernel: Array4<f32>) -> Self
    {
        Kernel{
            kernel
        }
    }
}
