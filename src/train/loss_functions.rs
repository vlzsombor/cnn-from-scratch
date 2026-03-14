use ndarray::Array1;

pub fn softmax(z: Array1<f32>) -> Array1<f32> {
    let max = z.fold(f32::NEG_INFINITY, |acc, &e| acc.max(e));
    let exp = z.mapv(|x| f32::exp(x - max));
    let sum = exp.sum();
    exp / sum
}



#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};
    use crate::train::loss_functions::softmax;

    #[test]
    pub fn test1()
    {
        let input :Array1<f32>= array![1,3,2].map(|&x| x as f32);
        let expected :Array1<f32>= array![0.090032,0.6652,0.244728].map(|&x| x as f32);
        let result = softmax(input);
        assert_abs_diff_eq!(&expected, &result, epsilon = 1e-4);
    }
}