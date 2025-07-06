use rust_annie::metrics::{euclidean, cosine, manhattan, chebyshev};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn test_euclidean() {
    let a = &[1.0, 2.0, 3.0];
    let b = &[4.0, 5.0, 6.0];
    let dist = euclidean(a, b);
    assert!(approx_eq(dist, 5.196152, 1e-5));
}

#[test]
fn test_cosine() {
    let a = &[1.0, 0.0];
    let b = &[0.0, 1.0];
    let dist = cosine(a, b);
    assert!(approx_eq(dist, 1.0, 1e-5)); // cosine distance of orthogonal vectors is 1
}

#[test]
fn test_manhattan() {
    let a = &[1.0, 2.0, 3.0];
    let b = &[4.0, 5.0, 6.0];
    let dist = manhattan(a, b);
    assert!(approx_eq(dist, 9.0, 1e-5));
}

#[test]
fn test_chebyshev() {
    let a = &[1.0, 2.0, 3.0];
    let b = &[4.0, 5.0, 7.0];
    let dist = chebyshev(a, b);
    assert!(approx_eq(dist, 4.0, 1e-5));
}
