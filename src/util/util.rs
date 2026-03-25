use crate::train::loss_functions::softmax;
use ndarray::{Array2, Axis};

pub fn one_hot(labels: &[usize], num_classes: usize) -> Array2<f32>
{
    let mut arr = Array2::zeros((labels.len(), num_classes));
    for (i, &label) in labels.iter().enumerate(){
        arr[[i,label]] = 1.0;
    }
    arr
}

pub fn normalize_features(x: &Array2<f32>) -> Array2<f32> {
    let mean = x.mean_axis(Axis(0)).unwrap();
    dbg!(&mean);
    let std = x.std_axis(Axis(0), 0.0);
    (x - &mean) / &std
}
pub fn accuracy(logits: &Array2<f32>, y: &Array2<f32>) -> f32 {
    let batch = logits.shape()[0];
    let a = softmax(logits.view());
    let pred = a.map_axis(Axis(1), |row| {
        row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0 as f32
    });
    let true_labels = y.map_axis(Axis(1), |row| {
        row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0 as f32
    });
//    dbg!(&pred, &a.row(0), &a.row(1), &a.row(2));

    let correct = pred.iter().zip(true_labels.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f32 / batch as f32
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
