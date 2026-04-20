use ndarray::{s, Array4};
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::CnnSigmoidActivation::CnnSigmoidActivation;
use crate::train::convolutional::convolutional_matlab::{ConvolutionalMatlab};
use crate::train::convolutional::ImageData::ImageData;

pub struct ConvolutionalContainer{
    pub layers: Vec<Box<dyn CnnLayerable>>
}

impl ConvolutionalContainer{
    pub fn new_default() -> Self{
        let alpha = 0.0001;
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 3, 3, 3));
            k.slice_mut(s![..,..,1,1]).fill(1.0);
            k
        };
        let layers: Vec<Box<dyn CnnLayerable>> = vec![
            Box::new(ConvolutionalMatlab::new(kernel, alpha)),
            Box::new(CnnSigmoidActivation::new())
        ];
        ConvolutionalContainer{
            layers
        }
    }
    pub fn forward(&mut self, image_data: ImageData) -> ImageData{
        self.layers
            .iter_mut()
            .fold(image_data, |acc, layer|{
                ImageData::new(layer.forward_propagation(&acc))
            })
    }

    pub fn backward(&mut self, image_data: ImageData) -> ImageData {
        let result = self.layers
            .iter_mut()
            .rev()
            .fold(image_data, |acc, layer|{
                let data = layer.backward_propagation(&acc);
                ImageData::new(data)
            });
        result
    }
}