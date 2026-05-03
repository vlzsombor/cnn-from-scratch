use crate::train::convolutional::CnnLayerable::CnnLayerable;
use crate::train::convolutional::CnnPoolingLayer::CnnPoolingLayer;
use crate::train::convolutional::CnnSigmoidActivation::CnnSigmoidActivation;
use crate::train::convolutional::ImageData::ImageData;
use crate::train::convolutional::convolutional_matlab::ConvolutionalMatlab;
use crate::train::layer::{Activation, ActivationLayer, Layer, xavier};
use crate::train::layer_container::LayerContainer;
use crate::train::layerable::Layerable;
use crate::util::ndarray_helper;
use crate::util::ndarray_helper::xavier2;
use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, Dim, Ix1, OwnedRepr, Shape, s};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CnnContainer {
    pub layers: Vec<Box<dyn CnnLayerable>>,
    pub linear_container: LayerContainer,
    image_data: Option<ImageData>,
}

impl CnnContainer {
    pub fn new_default(learning_rate: f32) -> Self {
        let alpha = learning_rate;
        let kernel: Array4<f32> = {
            let mut k = xavier2(&[1, 6, 5, 5])
                .into_dimensionality::<ndarray::Ix4>()
                .expect("kernel initialization failed");
            // k.slice_mut(s![..,..,1,1]).fill(1.0);
            k
        };

        let kernel2: Array4<f32> = {
            let mut k = xavier2(&[6, 12, 5, 5])
                .into_dimensionality::<ndarray::Ix4>()
                .expect("kernel initialization failed");
            k
        };

        let layers: Vec<Box<dyn CnnLayerable>> = vec![
            Box::new(ConvolutionalMatlab::new(kernel, alpha)),
            Box::new(CnnSigmoidActivation::new()),
            Box::new(CnnPoolingLayer::new()),
            Box::new(ConvolutionalMatlab::new(kernel2, alpha)),
            Box::new(CnnSigmoidActivation::new()),
            Box::new(CnnPoolingLayer::new()),
        ];
        let linear_layers: Vec<Box<dyn Layerable>> = vec![
            Box::new(Layer::new(192, 10, alpha)),
            //            Box::new(ActivationLayer::sigmoid()),
        ];
        let linear_container = LayerContainer::new_layers_boxed(linear_layers);
        CnnContainer {
            layers,
            linear_container,
            image_data: None,
        }
    }
    pub fn forward(&mut self, image_data: &ImageData) -> Array2<f32> {
        let cnn_res = self
            .layers
            .iter_mut()
            .fold(image_data.clone(), |acc, layer| {
                ImageData::new(layer.forward_propagation(&acc))
            });
        self.image_data = Some(cnn_res.clone());
        let array2 = cnn_res
            .image
            .view()
            .into_shape_with_order((1, cnn_res.image.len()))
            .expect("reshape failed")
            .to_owned();
        let r = self.linear_container.forward(&array2);
        r
    }

    pub fn backward_linear(&mut self, delta: Array2<f32>) -> ImageData {
        let back_res = self.linear_container.backward_propagation(delta);
        let image_data = self.image_data.as_ref().unwrap().image.shape();
        let restored: Array3<f32> = back_res
            .into_shape_with_order((image_data[0], image_data[1], image_data[2]))
            .unwrap();
        self.backward(ImageData::new(restored))
    }
    pub fn backward(&mut self, image_data: ImageData) -> ImageData {
        let result = self.layers.iter_mut().rev().fold(image_data, |acc, layer| {
            let data = layer.backward_propagation(&acc);
            ImageData::new(data)
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::train::convolutional::CnnContainer::CnnContainer;
    use crate::train::convolutional::ImageData::ImageData;
    use crate::train::layer_container::mse_loss_derivative;
    use crate::util::mnist_helper::load_mnist;
    use ndarray::{Array2, Array3, array};

    #[test]
    pub fn test_default_network() {
        let mut sut = CnnContainer::new_default(0.001);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
        let first = x.row(0); //.unwrap();
        let first = first.mapv(|x| x / 255.);
        let first = first.view();
        let two_dimensional_data: Array2<f32> =
            first.into_shape_with_order((28, 28)).unwrap().to_owned();

        let image_data: Array3<f32> = first.to_shape((1, 28, 28)).unwrap().to_owned();
        dbg!(&image_data.shape());
        let image_data_wrapper = ImageData::new(image_data);

        let target: Array2<f32> = array![[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,]];

        let mut y_hat = sut.forward(&image_data_wrapper);
        for _ in 0..100 {
            y_hat = sut.forward(&image_data_wrapper);
            let a = mse_loss_derivative(&y_hat, &target);
            dbg!(&a);
            sut.backward_linear(a);
        }
        let a = sut.backward_linear(target);
    }
}
