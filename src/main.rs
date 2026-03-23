extern crate core;

use crate::train::layer::{Activation, ActivationLayer, Layer};
use crate::train::layer_container::{loss, loss_derivate, LayerContainer};
use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod input_transform;
pub mod train;

fn generate_xor_dataset() -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = array![[0.,0.],
                                [0.,1.],
                                [1.,0.],
                                [1.,1.],
    ];
    // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y = array![[0., 1.], [1., 0.], [1., 0.],[0., 1.]];
    (x, y)
}
fn generate_linear_dataset(n_samples: usize) -> (Array2<f32>, Array2<f32>) {
    let x: Array2<f32> = Array2::random((n_samples, 2), StandardNormal) * 2. + 10.;
   // let y: Array1<f32> = 2.0 * &x.column(0) + 3.0 * &x.column(1) - 1.0;
    let y: Array1<f32> = x.column(0).to_owned() + x.column(0).to_owned();
    (x, y.insert_axis(Axis(1))
        .to_owned())
}


pub fn debug_array(a: &Array2<f32>) {
    let rows: Vec<String> = a.rows().into_iter()
        .map(|row| {
            let vals: Vec<String> = row.iter().map(|x| format!("{:.3}", x)).collect();
            format!("[{}]", vals.join(", "))
        })
        .collect();
    println!("[{}]", rows.join(", "));
}

fn main() {
    let layers: Vec<Box<dyn crate::train::layerable::Layerable>> = vec![
        Box::new(Layer::new(2, 64)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(64, 64)),
        Box::new(ActivationLayer::relu()),
        Box::new(Layer::new(64, 2)),
    ];
    let mut sut = LayerContainer::new_layers_boxed(layers);

    let (X, y) = generate_xor_dataset();

    let mut y_hat = sut.forward(&X);
    let l1 = loss(&y_hat, &y);
    for i in 0..10000{
        y_hat = sut.forward(&X);
        sut.backward_propagation(loss_derivate(&y_hat, &y));
        if i % 1000 == 0 {
            let l = loss(&y_hat, &y);
            dbg!(i, l);
        }
    }
    let l2 = loss(&y_hat, &y);

    let xp = array![[0.,0.], [0., 1.], [1.,0.], [1.,1.]];
    let predict1 = sut.forward(&xp);
    debug_array(&xp);
    debug_array(&predict1);
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