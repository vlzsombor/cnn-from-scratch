use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::input_transform::{normalize_image_pixels_vec, process_csv};
use crate::train::LayerContainer::{loss, loss_derivate, LayerContainer};
use ndarray::{array, Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use crate::train::layer::{Activation, Layer};

mod input_transform;
pub mod train;

const U8_UPPER: u32 = 255;
fn generate_linear_dataset(n_samples: usize) -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = Array2::random((n_samples, 2), StandardNormal) * 2. + 10.;
   // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y: Array1<f32> = x.column(0).to_owned() + x.column(0).to_owned();
    (x, y.insert_axis(Axis(1))
        .to_owned())
}
fn main() {
    let layers: Vec<Layer> = vec![
        Layer::new(2,64, Some(Activation::Relu()),None),
        Layer::new(64,1, Some(Activation::Relu()),None),
    ];
    let mut sut = LayerContainer::new_layers(layers);

    let (X, y) = generate_linear_dataset(1000);

    dbg!(&X);
    let mut y_hat = sut.forward(&X);
    let l1 = loss(&y_hat, &y);
    for i in 0..1500{
        y_hat = sut.forward(&X);
        sut.backward_hard_coded(&loss_derivate(&y_hat, &y));
        if(i % 50 == 0){
            let l = loss(&y_hat, &y);
            dbg!(l);
        }
    }
    let l2 = loss(&y_hat, &y);

    let xp = array![[10.,10.2], [10., 10.3], [10.,9.9], [10.,9.8], [11.,9.5], [10.2,10.8], [10.,9.], [10.5,9.5]];
    let predict1 = sut.forward(&xp);
    dbg!(&xp);
    dbg!(&predict1);
    dbg!(l2);
    return;






    let xor_input = array![[0.,0.], [0.,1.],[1.,0.], [1.,1.]];
    let y = array![[0.,1.], [1.,0.],[1.,0.], [0.,1.]];
    // [[1, 2, 3],
    //  [4, 5, 6]]
    let layers = vec![
        Layer::new(2,2, Some(Activation::Relu()),None),
        Layer::new(2,2, Some(Activation::Relu()),None)
    ];
    let mut xor_container = LayerContainer::new_layers(layers);
    let y_hat = xor_container.forward(&xor_input);
    assert_eq!(y_hat.len(), y.len());
    let n = y_hat.len() as f32;
    let loss: Array2<f32> = 1./n * (y_hat - y);

    xor_container.backward_hard_coded(&loss);

}

fn train(data: &Vec<(Vec<f32>, String)>) -> Option<i32>
{
    let x: Vec<f32> = data.iter()
        .map(|(x, _)| x)
        .flatten()
        .copied()
        .collect();

    let label: Vec<f32> = data
        .iter()
        .map(|(_, y)| {
            if y == "Iris-setosa" { 0. } else { 1. }
        })
        .collect();


    let mut node_container = LayerContainer::new_def();
    let matrix_f32: Array2<f32> = Array2::from_shape_vec(
        (150, 4),
        x
    ).unwrap();
    let y = Array1::from(label);

    for i in 0..3 {
        let y_hats = node_container.forward(&matrix_f32);
        let col: Array1<f32> = y_hats.column(0).to_owned();
        let a1 = col - &y;
        let grad: Array2<f32> = a1.insert_axis(Axis(0)); // shape: (1, 3)
        let b = node_container.backward2(grad);
    }
    Some(0)
}

fn load_iris(path: &str) -> Result<Vec<(Vec<f32>, String)>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 6 {
            let class = parts[5];
            let features: Vec<f32> = parts[1..5]
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();

            data.push((features, class.to_string()));
        }
    }

    if data.is_empty() {
        return Err("Keine Daten gefunden".into());
    }

    Ok(data)
}
