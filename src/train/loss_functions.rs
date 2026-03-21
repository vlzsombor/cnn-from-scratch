use ndarray::Array1;

pub const EPSILON: f32 = 1.0E-6;//0.0001;

pub fn softmax(z: Array1<f32>) -> Array1<f32> {
    let max = z.fold(f32::NEG_INFINITY, |acc, &e| acc.max(e));
    let exp = z.mapv(|x| f32::exp(x - max));
    let sum = exp.sum();
    exp / sum
}

pub fn cross_entropy_loss(y: Array1<f32>, y_hat: Array1<f32>) -> f32 {
    -(y * y_hat.mapv(|p| (p + EPSILON).ln())).sum()
}


#[cfg(test)]
mod tests {
    use crate::train::loss_functions::{cross_entropy_loss, softmax, EPSILON};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    #[test]
    pub fn test1()
    {
        let input :Array1<f32>= array![1,3,2].map(|&x| x as f32);
        let expected :Array1<f32>= array![0.090032,0.6652,0.244728].map(|&x| x as f32);
        let result = softmax(input);
        assert_abs_diff_eq!(&expected, &result, epsilon = 1e-4);
    }

    #[test]
    pub fn cross_entropy_loss_test()
    {
        let y_hat :Array1<f32>= array![0,0,1].map(|&x| x as f32);
        let y :Array1<f32>= array![1,0,0].map(|&x| x as f32);
        let result = cross_entropy_loss(y, y_hat);
        assert_abs_diff_eq!(&13.815511, &result, epsilon = 1e-4);

        let y = array![1.0, 0.0, 0.0];
        let y_hat = array![0.9, 0.05, 0.05];
        let result = cross_entropy_loss(y, y_hat);
        assert_abs_diff_eq!(-f32::ln(0.9), result, epsilon = 1e-5); // 0.1

        let y = array![1.0, 0.0, 0.0];
        let y_hat = array![0.2, 0.95, 0.05];
        let result = cross_entropy_loss(y, y_hat);
        assert_abs_diff_eq!(-f32::ln(0.2), result, epsilon = 1e-5); // 1.6


        let y = array![1.0, 0.0, 0.0];
        let y_hat = array![1., 0., 0.];
        let result = cross_entropy_loss(y, y_hat);
        assert_abs_diff_eq!(-f32::ln(1.), result, epsilon = 1e-5); // 0.1

    }


}