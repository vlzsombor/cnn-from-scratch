use ndarray::{Array1, Array2};
use crate::train::layer::Layer;

struct node_container{
    Layers: Vec<Layer>
}

impl node_container{

    pub fn new(layer_number: Vec<[u32; 2]>) -> Self {
        let layers: Vec<Layer> = layer_number.iter().map(|x|{
            let a = x[0];
            let b = x[1];
            Layer::new(a,b, None, None)
        }).collect();
        node_container{
            Layers: layers
        }
    }

    pub fn forward(&self, X: Array2<f32>) -> Array2<f32>{
//        self.Layers
//            .iter()
//            .for_each(|x|{
//                x.forward()
//            });
        self.Layers
            .iter()
            .fold(X, |acc, layer| {
                layer.forward(&acc)
            })
    }
}


#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use crate::train::node_container::node_container;

    #[test]
    pub fn test1()
    {
        let layer_number = vec![[3,2],[2,1]];
        let sut = node_container::new(layer_number);
        let input: Array2<f32> = Array2::from(vec![[1.,2.,3.], [5.,6.,7.], [9.,10.,11.], [20.,300.,200000.]]);
        let input: Array2<f32> = Array2::from(vec![[10.,0., 10.], [100000.,10000., 100000.]]);
//        let input: Array2<f32> =  Array2::from(vec![[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,20.]]);
        let res = sut.forward(input);
        println!("res \n{:#?}", res);

    }
}