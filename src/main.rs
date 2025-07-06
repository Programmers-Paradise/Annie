#[cfg(any(feature = "cuda", feature = "rocm"))]
use annie::gpu::gpu::l2_distance_gpu;

fn main() {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        let dim = 3;
        let n_queries = 2;
        let n_vectors = 2;
        let queries = vec![
            1.0, 2.0, 3.0, // Query 1
            4.0, 5.0, 6.0, // Query 2
        ];
        let corpus = vec![
            1.0, 0.0, 0.0, // Vector 1
            0.0, 1.0, 0.0, // Vector 2
        ];
        
        let distances = l2_distance_gpu(&queries, &corpus, dim, n_queries, n_vectors);
        println!("GPU L2 distances: {:?}", distances);
    }

    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    println!("Run with --features cuda or --features rocm to enable GPU support.");
}
