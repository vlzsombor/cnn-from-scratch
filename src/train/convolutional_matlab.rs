use crate::train::activation::ReluActivation;
use crate::train::layerable::Layerable;
use ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2};
use std::fmt::Debug;
pub const ALPHA: f32 = 0.00001;

#[derive(Debug, Clone)]
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
#[derive(Debug)]
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
#[derive(Debug)]
pub struct ConvolutionalMatlab
{
    kernel: Kernel,
    biases: Array1<f32>,
    alpha: f32,
    cached_input: Option<ImageData>
}

impl ConvolutionalMatlab
{
    pub fn compute_mse(output: &Array3<f32>, target: &Array3<f32>) -> f32 {
        let diff = output - target;
        0.5 * diff.mapv(|x| x.powi(2)).sum()
    }
    pub fn new(kernel: Array4<f32>, alpha: f32) -> Self
    {
        let k = Kernel::new(kernel);
        let biases = Array1::zeros(k.get_output_channel_number());// bug in channels should be out channels
        ConvolutionalMatlab
        {
            kernel: k,
            biases,
            alpha,
            cached_input: None
        }
    }
    pub fn forward_activation(&mut self, x: &ImageData) -> Array3<f32>
    {
        self.forward(x).get_sigmoid()
    }
    pub fn forward(&mut self, x: &ImageData) -> Array3<f32>
    {
        assert_eq!(x.get_channel_number(), self.kernel.get_input_channel_number());
        let (new_row, new_col) = x.get_image_size_after_convolution(&self.kernel);
        let mut return_res = x.get_new_temp_array(&self.kernel); //Array3::zeros([self.kernel.get_output_channel_number(), new_row, new_col]);
        for output_index in 0..self.kernel.get_output_channel_number(){
            let mut acc = Array2::zeros((return_res.shape()[1], return_res.shape()[2]));
            for input_index in 0..self.kernel.get_input_channel_number() {
                let sub_x = x.get_image(input_index);
                let sub_k = self.kernel.get_image(input_index, output_index);
                let r =  Self::conv2d(&sub_x, &sub_k);
                acc = r + acc;
            }
            acc = acc + self.biases[output_index];
            return_res
                .slice_mut(s![output_index, ..,..])
                .assign(&acc);
        }
        // dbg!(&self.biases.shape(),&return_res.shape());
        // let b = return_res + &self.biases;
        self.cached_input = Some(x.clone());
        return_res
    }
    pub fn k1_back(&mut self, delta_c_p: &ImageData) {
        let grad = self.compute_kernel_gradient(delta_c_p);
        self.apply_kernel_update(&grad);
    }
    pub fn compute_kernel_gradient(&self, delta_c_p: &ImageData) -> Array4<f32> {
        let (new_r, new_c) = self.cached_input.as_ref().unwrap()
            .get_image_size_after_convolution_image_data(delta_c_p);
        let mut grad = Array4::zeros([
            self.kernel.get_input_channel_number(),
            self.kernel.get_output_channel_number(),
            new_r, new_c
        ]);
        let cached = self.cached_input.as_ref().unwrap();
        for out_idx in 0..self.kernel.get_output_channel_number() {
            for in_idx in 0..self.kernel.get_input_channel_number() {
                let sub_image  = cached.get_image(in_idx);
                let sub_delta  = delta_c_p.get_image(out_idx);
                let a = Self::conv2d(&sub_image, &sub_delta);
                grad.slice_mut(s![in_idx, out_idx, .., ..]).assign(&a);
            }
        }
        grad
    }

    pub fn apply_kernel_update(&mut self, grad: &Array4<f32>) {
        self.kernel.subtract(&(self.alpha * grad));
    }
    pub fn conv2d(data: &ArrayView2<f32>, kernel: &ArrayView2<f32>) -> Array2<f32> {
        let new_row = data.shape()[0] - kernel.shape()[0] + 1;
        let new_col = data.shape()[1] - kernel.shape()[1] + 1;
        let mut return_data: Array2<f32> = Array2::zeros((new_row, new_col));
        let k_row = kernel.shape()[0];
        let k_col = kernel.shape()[1];
        for i in 0..new_row {
            for j in 0..new_col {
                let sub_data = &data.slice(s![i..i+k_row, j..j+k_col]);
                let multiply = sub_data * kernel;
                return_data[[i,j]] = multiply.sum()
            }
        }
        return_data
    }
}
#[cfg(test)]
mod tests {
    use crate::train::activation::ReluActivation;
    use crate::train::convolutional_matlab::{ConvolutionalMatlab, ImageData, ALPHA};
    use crate::util::mnist_helper::load_mnist;
    use ndarray::{array, s, Array1, Array2, Array3, Array4, ArrayView2};
    #[test]
    pub fn c1_gradient_check2() {}

    #[test]
    pub fn c1_gradient_check() {
        let epsilon = 1e-3_f32;
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 6, 5, 5));
            k.slice_mut(s![.., .., 2, 2]).fill(1.0);
            k
        };
        let (x, _y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
        let input_image: Array3<f32> = x.row(0)
            .into_shape_with_order((1, 28, 28)).unwrap().mapv(|x| x / 255.).to_owned();
        let image_data = ImageData::new(input_image);
        let target: Array3<f32> = Array3::ones((6, 24, 24));

        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);

        let output = sut.forward(&image_data);              // cacht input
        let grad_output = &output - &target;           // ∂L/∂output, [6,24,24]
        let delta = ImageData::new(grad_output.clone());
        let analytical_grad: Array4<f32> = sut.compute_kernel_gradient(&delta);

        let mut sut_plus = ConvolutionalMatlab::new({
                                                        let mut k = Array4::zeros((1, 6, 5, 5));
                                                        k.slice_mut(s![.., .., 2, 2]).fill(1.0);
                                                        k[[0, 0, 2, 2]] += epsilon;
                                                        k
                                                    }, ALPHA);
        let out_plus = sut_plus.forward(&image_data);
        let loss_plus = ConvolutionalMatlab::compute_mse(
            &out_plus, &target
        );

        let mut sut_minus = ConvolutionalMatlab::new({
                                                         let mut k = Array4::zeros((1, 6, 5, 5));
                                                         k.slice_mut(s![.., .., 2, 2]).fill(1.0);
                                                         k[[0, 0, 2, 2]] -= epsilon;
                                                         k
                                                     }, ALPHA);
        let out_minus = sut_minus.forward(&image_data);
        let loss_minus = ConvolutionalMatlab::compute_mse(
            &out_minus, &target
        );

        let numerical_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        let analytic = analytical_grad[[0, 0, 2, 2]];

        println!("numerical: {numerical_grad:.6},  analytical: {analytic:.6}");
        let relative_error = (analytic - numerical_grad).abs() / analytic.abs().max(numerical_grad.abs());
        assert!(relative_error < 1e-3,
                "Gradient check failed: {numerical_grad} vs {analytic} relative: {relative_error}");
    }

    #[test]
    pub fn c1_test()
    {
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 6, 5, 5));
            k.slice_mut(s![.., .., 2, 2]).fill(1.0);
            k
        };
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let input_image: Array3<f32> = first.into_shape_with_order((1, 28, 28))
            .unwrap()
            .to_owned();
        let imageData = ImageData::new(input_image);
        let mut sut = ConvolutionalMatlab::new(kernel, 0.0001);
        let f_res = sut.forward(&imageData);
        assert_eq!(f_res.shape(), [6, 24, 24]);

        let f = &ImageData::new(f_res);
        let r = sut.compute_kernel_gradient(&f);
        assert_eq!(r.shape(), [1, 6, 5, 5]);
    }

    #[test]
    pub fn c1_back_prop_test()
    {
        let kernel: Array4<f32> = array![
            [[
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
            ]]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28, 28))
            .unwrap()
            .to_owned();

        let a = ImageData::new(a);
        let res = sut.forward(&a);
        let y: Array3<f32> = Array3::ones((1, 24, 24)) * 250.;
        for _ in 0..100 {
            let y_hat = sut.forward(&a);
            let delta_y_hat = (&y_hat - &y) * &y_hat.get_sigmoid_derivative();
            //            let res = sut.k1_back(&delta_y_hat);

            //           let dummy_loss = res.mapv(|x| x.powi(2)).sum();
        }
    }
    #[test]
    pub fn c1_size_test2()
    {
        let kernel: Array4<f32> = array![
            [[
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
            ]]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28, 28))
            .unwrap()
            .to_owned();

        let a = ImageData::new(a);
        let res = sut.forward(&a);

        let aslice: &ArrayView2<f32> = &a.image.slice(s![0, 1..25,1..25]);
        let ressss: &ArrayView2<f32> = &res.slice(s![0,..,..]);

        let final_res = &a.image.slice(s![0, 16, 2..26]);
        let final_res2 = &ressss.row(14);
        assert_eq!(&final_res, &final_res2);
        //    assert_eq!(&aslice.shape(), &ressss.shape());
        assert_eq!(&aslice.row(0), &ressss.row(0));
    }
    #[test]
    pub fn c1_size_test()
    {
        let kernel: Array4<f32> = Array4::ones((1, 6, 5, 5));
        let mut sut = ConvolutionalMatlab::new(kernel, ALPHA);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();
        let first = x.row(0); //.unwrap();
        let a: Array3<f32> = first.into_shape_with_order((1, 28, 28))
            .unwrap()
            .to_owned();
        let res = sut.forward(&ImageData::new(a));
        assert_eq!(res.shape(), &[6, 24, 24]);
    }

    #[test]
    pub fn conv2d_test() {
        let kernel: Array2<f32> = array![
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,1.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0],
                [0.0,0.0,0.0,0.0,0.0]
        ];
        let biases: Array1<f32> = Array1::ones(kernel.shape()[0]);
        let (x, y) = load_mnist("src/data/mnist_train_small.csv").unwrap();

        let first = x.row(0); //.unwrap();
        let a: Array2<f32> = first.into_shape_with_order((28, 28))
            .unwrap()
            .to_owned();

        let res = ConvolutionalMatlab::conv2d(&a.view(), &kernel.view());

        let aslice: &ArrayView2<f32> = &a.slice(s![1..25,1..25]);
        let ressss: &ArrayView2<f32> = &res.slice(s![..,..]);

        let final_res = &a.slice(s![16, 2..26]);
        let final_res2 = &ressss.row(14);
        assert_eq!(&final_res, &final_res2);
        //    assert_eq!(&aslice.shape(), &ressss.shape());
        assert_eq!(&aslice.row(0), &ressss.row(0));
    }
    #[test]
    pub fn backward_output_dimension() {
        let epsilon = 1e-3_f32;
        let alpha = 1.;
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 3, 3, 3));
            k.slice_mut(s![..,..,1,1]).fill(1.0);
            k
        };
        //1 5x5
        let input_image_raw: Array3<f32> = Array3::ones((1, 5, 5)) * 0.1;
        let input_image = ImageData::new(input_image_raw);

        let target_image_raw: Array3<f32> = Array3::ones((3, 3, 3)) * 0.5;
        let target_image = ImageData::new(target_image_raw);
        let mut sut = ConvolutionalMatlab::new(kernel.clone(), alpha);
        sut.forward(&input_image);
        let backward = sut.compute_kernel_gradient(&target_image);
        assert_eq!(backward.raw_dim(), kernel.raw_dim());
    }
    #[test]
    pub fn gradient_check_dummy_cnn() {
        //Test model: kernel 2,1,3x3
        let epsilon = 1e-3_f32;
        let alpha = 1.;
        let kernel: Array4<f32> = {
            let mut k = Array4::zeros((1, 3, 3, 3));
            k.slice_mut(s![..,..,1,1]).fill(1.0);
            k
        };
        //1 5x5
        let input_image_raw: Array3<f32> = Array3::ones((1, 5, 5)) * 0.1;
        let input_image = ImageData::new(input_image_raw);

        let target_image_raw: Array3<f32> = Array3::ones((3, 3, 3)) * 0.5;
        let target_image = ImageData::new(target_image_raw);
        let size = (&kernel.clone()).shape()[2];
        let in_size = (&kernel.clone()).shape()[0];
        let out_size = (&kernel.clone()).shape()[1];
        let mut i = 0;
        let mut sut = ConvolutionalMatlab::new(kernel.clone(), alpha);
        let f = sut.forward(&input_image);

        let delta = &f - &target_image.image;
        let analytic = sut.compute_kernel_gradient(&ImageData::new(delta));
        for inp in 0..in_size {
            for out in 0..out_size {
                for row in 0..size {
                    for col in 0..size {
                        let kernel = kernel.clone();
                        let idx = (inp, out, row, col);
                        let mut sut_plus = ConvolutionalMatlab::new({
                                                                        let mut k = kernel.clone();
                                                                        k[idx] += epsilon;
                                                                        k.clone()
                                                                    }, alpha);

                        let mut sut_minus = ConvolutionalMatlab::new({
                                                                         let mut k = (&kernel).clone();
                                                                         k[idx] -= epsilon;
                                                                         k
                                                                     }, alpha);

                        let f_plus = sut_plus.forward(&input_image);
                        let f_minus = sut_minus.forward(&input_image);
                        let loss_plus = ConvolutionalMatlab::compute_mse(
                            &f_plus, &target_image.image
                        );
                        let loss_minus = ConvolutionalMatlab::compute_mse(
                            &f_minus, &target_image.image
                        );
                        let numeric_grad = (loss_plus - loss_minus) / (2. * epsilon);


                        let analytic_grad = analytic[idx];
                        let relative_error = (&analytic_grad - &numeric_grad).abs() / &analytic_grad.abs().max(numeric_grad.abs());
                        assert!(relative_error < 1e-3,
                                "Gradient check failed: {analytic_grad} vs {numeric_grad} relative: {relative_error}");
                        dbg!(&relative_error, i);
                        i += 1;
                    }
                }
            }
        }
    }
}