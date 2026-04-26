pub fn xavier2(shape: &[usize]) -> ndarray::ArrayD<f32>{
    let fan_in: usize = shape[1..].iter().product();
    let fan_out: usize = shape[0] * shape[2..].iter().product::<usize>();
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    ndarray::ArrayD::from_shape_fn(shape, |_|{
        rand::random::<f32>() * 2.0 * limit - limit
    })
}
