#![no_main]
use libfuzzer_sys::fuzz_target;
use rust_annie::metrics::{euclidean, cosine, manhattan, chebyshev};

fuzz_target!(|data: &[u8]| {
    // Split input into two vectors of f32
    let (a, b) = if data.len() < 8 {
        return; // Need at least 8 bytes (2 floats)
    } else {
        let mid = data.len() / 2;
        let (left, right) = data.split_at(mid);
        (
            bytes_to_f32_vec(left),
            bytes_to_f32_vec(right)
        )
    };

    // Ensure vectors are the same length
    let len = std::cmp::min(a.len(), b.len());
    if len == 0 {
        return;
    }
    let a = &a[..len];
    let b = &b[..len];

    // Call all distance functions
    let _ = euclidean(a, b);
    let _ = cosine(a, b);
    let _ = manhattan(a, b);
    let _ = chebyshev(a, b);
});

fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            f32::from_ne_bytes(bytes)
        })
        .filter(|&f| {
            // Filter out signaling NaNs but allow quiet NaNs
            !(f.is_nan() && (f.to_bits() & 0x00400000 == 0))
        })
        })
        .collect()
}