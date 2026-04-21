use ndarray::{s, Array1, Array2, Array4, ArrayBase, Ix1, OwnedRepr};
use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::CnnPoolingLayer::CnnPoolingLayer;
use crate::train::convolutional::CnnSigmoidActivation::CnnSigmoidActivation;
use crate::train::convolutional::convolutional_matlab::{ConvolutionalMatlab};
use crate::train::convolutional::ImageData::ImageData;
use crate::train::layer::Layer;
use crate::train::layer_container::LayerContainer;
use crate::train::layerable::Layerable;

pub struct CnnContainer {
    pub layers: Vec<Box<dyn CnnLayerable>>,
    pub linear_container: LayerContainer
}

impl CnnContainer {
    pub fn new_default() -> Self{
        let alpha = 0.0001;
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 6, 5, 5));
            k.slice_mut(s![..,..,1,1]).fill(1.0);
            k
        };
        let layers: Vec<Box<dyn CnnLayerable>> = vec![
            Box::new(ConvolutionalMatlab::new(kernel, alpha)),
            Box::new(CnnSigmoidActivation::new()),
            Box::new(CnnPoolingLayer::new()),
        ];
        let linear_layers: Vec<Box<dyn Layerable>> = vec![
            Box::new(Layer::new(864, 64, 0.001)),
            Box::new(Layer::new(64, 10, 0.001))
        ];
        let linear_container = LayerContainer::new_layers_boxed(linear_layers);
        CnnContainer {
            layers,
            linear_container
        }
    }
    pub fn forward(&mut self, image_data: ImageData) -> Array2<f32> {
        let cnn_res = self.layers
            .iter_mut()
            .fold(image_data, |acc, layer|{
                ImageData::new(layer.forward_propagation(&acc))
            });
        let vectorized = CnnContainer::vectorization(&cnn_res);
        let len = vectorized.len();
        let array2 = vectorized.view().into_shape_with_order((1,len)).expect("reshape failed").to_owned();
        self.linear_container.forward(&array2)
    }

    pub fn vectorization(image_data: &ImageData) -> Array1<f32> {
        let ch_size = image_data.get_channel_number();
        let row_size = image_data.get_row();
        let col_size = image_data.get_col();
        let mut vectorized = Array1::zeros(ch_size * row_size * col_size);
        for channel in 0..image_data.get_channel_number() {
            for row in 0..image_data.get_row(){
                for col in 0..image_data.get_col(){
                    vectorized[channel * ch_size + row * row_size + col * col_size ] = image_data.image[[channel, row, col]];
                }
            }
        }
        vectorized
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

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array3};
    use crate::train::convolutional::CnnContainer::CnnContainer;
    use crate::train::convolutional::ImageData::ImageData;
    use crate::util::mnist_helper::load_mnist;

    #[test]
    pub fn test_default_network() {
        let mut sut = CnnContainer::new_default();

        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let two_dimensional_data: Array2<f32> = first.into_shape_with_order((28, 28))
            .unwrap()
            .to_owned();

        let image_data: Array3<f32> = first.to_shape((1, 28, 28)).unwrap().to_owned();
        dbg!(&image_data.shape());
        let a = sut.forward(ImageData::new(image_data));
        dbg!(&a);
//        dbg!(&a.image.shape());
    }
}
