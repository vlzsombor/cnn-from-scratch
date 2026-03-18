use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::input_transform::{normalize_image_pixels_vec, process_csv};
use crate::train::LayerContainer::{loss, loss_derivate, LayerContainer};
use ndarray::{array, Array, Array1, Array2, Axis};
use crate::train::layer::{Activation, Layer};

mod input_transform;
pub mod train;

const U8_UPPER: u32 = 255;

fn main() {
//    let r = process_csv("src/data/mnist_train_small.csv").unwrap();
//    let normalized = normalize_image_pixels_vec(&r, U8_UPPER);

//    let r = load_iris("src/data/Iris.csv").unwrap();
//    let b = train(&r);





    let layers = vec![
        Layer::new(2,2, Some(Activation::Relu()),None),
        //            Layer::new(2,1, Some(Activation::Relu()),None),
    ];
    let mut sut = LayerContainer::new_layers(layers);
    let input = array![[1.,1.], [5.,5.]];
    let y = array![[1.,3.], [8.,6.],[48.,11.], [224.,19.]];
    let y = array![[2.,2.], [5., 5.]];

    let mut y_hat = sut.forward(&input);
    //        let loss: Array2<f32> = 1./(y_hat.nrows() as f32) * (y_hat - y);
    let l1 = loss(&y_hat, &y);
    for _ in 0..10{
        y_hat = sut.forward(&input);
        sut.backward_hard_coded(&loss_derivate(&y_hat, &y));
    }
    let l2 = loss(&y_hat, &y);
    dbg!(l1);
    dbg!(l2);




    return;


    let epsilon = 1e-5;






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
