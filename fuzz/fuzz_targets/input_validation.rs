use libfuzzer_sys::fuzz_target;
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;

fuzz_target!(|data: &[u8]| {
    // Fuzz input validation for AnnIndex
    if data.len() < 4 {
        return;
    }
    let dim = (data[0] as usize % 32) + 1; // dimension between 1 and 32
    let metric = match data[1] % 4 {
        0 => Distance::Euclidean(),
        1 => Distance::Cosine(),
        2 => Distance::Manhattan(),
        _ => Distance::Chebyshev(),
    };
    let index = AnnIndex::new(dim, metric);
    // Try adding random data
    if let Ok(mut idx) = index {
        let n = (data[2] as usize % 8) + 1;
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let start = 3 + i * dim;
            let end = start + dim;
            if end > data.len() { break; }
            let v: Vec<f32> = data[start..end].iter().map(|b| *b as f32).collect();
            vecs.push(v);
        }
        }
        let ids: Vec<usize> = (0..vecs.len()).collect();
        if !vecs.is_empty() && vecs.len() == ids.len() {
            // Ignore errors, just check for panics/crashes
            let _ = idx.add_vecs(&vecs, &ids);
        }
    }
});
