use crate::train::layer::{Activation, Layer};
use crate::train::layer_container::{loss, loss_derivate, LayerContainer};
use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod input_transform;
pub mod train;

fn generate_linear_dataset(n_samples: usize) -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = Array2::random((n_samples, 2), StandardNormal) * 2. + 10.;
   // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y: Array1<f32> = x.column(0).to_owned() + x.column(0).to_owned();
    (x, y.insert_axis(Axis(1))
        .to_owned())
}
fn main() {
    let layers: Vec<Layer> = vec![
        Layer::new(2, 64, Some(Activation::relu()), None),
        Layer::new(64, 1, Some(Activation::relu()), None),
    ];
    let mut sut = LayerContainer::new_layers(layers);

    let (X, y) = generate_linear_dataset(1000);

    dbg!(&X);
    let mut y_hat = sut.forward(&X);
    let l1 = loss(&y_hat, &y);
    for i in 0..1500{
        y_hat = sut.forward(&X);
        sut.backward_hard_coded(&loss_derivate(&y_hat, &y));
        if i % 50 == 0 {
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
