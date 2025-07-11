//new added
// use crate::distance::Distance;
use crate::metrics::Distance;



pub trait AnnBackend {
    fn new(dims: usize, distance: Distance) -> Self
    where
        Self: Sized;

    fn add_item(&mut self, item: Vec<f32>);
    fn build(&mut self);
    fn search(&self, vector: &[f32], k: usize) -> Vec<usize>;

    fn save(&self, path: &str);
    
    #[allow(dead_code)]
    fn load(path: &str) -> Self
    where
        Self: Sized;
}
