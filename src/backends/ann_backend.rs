/// Trait that defines a common interface for ANN backends.
/// Both BruteForceIndex and HnswIndex will implement this.
pub trait AnnBackend {
    /// Add a single vector to the index.
    fn add(&mut self, vector: Vec<f32>);

    /// Search for k nearest neighbors.
    /// Returns Vec of (index, distance).
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;

    /// Number of items stored.
    fn len(&self) -> usize;
}
