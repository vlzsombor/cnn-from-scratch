use ndarray::{Array1, Array2, ArrayView2, Axis};

pub const EPSILON: f32 = 1.0E-6;//0.0001;

pub fn softmax(x: ArrayView2<f32>) -> Array2<f32> {
    let result = x.map_axis(Axis(1), |row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max); // numerische Stabilität
        let exp: Array1<f32> = row.mapv(|xi| (xi - max).exp());
        let sum = exp.sum();
        exp / sum
    });
    to_array2(result)
}

fn to_array2(a: Array1<Array1<f32>>) -> Array2<f32> {
    let rows = a.len();
    let cols = a[0].len();
    let flat: Vec<f32> = a.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, cols), flat).unwrap()
}
pub fn cross_entropy_loss(y: Array1<f32>, y_hat: Array1<f32>) -> f32 {
    -(y * y_hat.mapv(|p| (p + EPSILON).ln())).sum()
}


#[cfg(test)]
mod tests {
    use crate::train::loss_functions::{cross_entropy_loss};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    pub fn test1()
    {
//        let input :Array1<f32>= array![1,3,2].map(|&x| x as f32);
//        let expected :Array1<f32>= array![0.090032,0.6652,0.244728].map(|&x| x as f32);
//        let result = softmax(input);
//        assert_abs_diff_eq!(&expected, &result, epsilon = 1e-4);
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