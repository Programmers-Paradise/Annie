use crate::metrics::Distance;
use crate::distance_registry::get_distance_function;

pub fn compute_distances_with_ids(
    entries: &[(i64, Vec<f32>, f32)],
    query: &[f32],
    query_sq_norm: f32,
    metric: Distance,
    minkowski_p: Option<f32>,
    k: usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut results: Vec<(i64, f32)> = entries
        .iter()
        .map(|(id, vec, vec_sq)| {
            let dist = if let Some(p) = minkowski_p {
                vec.iter()
                    .zip(query)
                    .map(|(x, y)| (x - y).abs().powf(p))
                    .sum::<f32>()
                    .powf(1.0 / p)
            } else {
                match &metric {
                    Distance::Euclidean() => ((vec_sq + query_sq_norm - 2.0 * dot(vec, query)).max(0.0)).sqrt(),
                    Distance::Cosine() => {
                        let denom = vec_sq.sqrt().max(1e-12) * query_sq_norm.sqrt().max(1e-12);
                        (1.0 - dot(vec, query) / denom).max(0.0)
                    }
                    Distance::Manhattan() => vec.iter().zip(query).map(|(x, y)| (x - y).abs()).sum(),
                    Distance::Chebyshev() => vec.iter().zip(query).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max),
                    Distance::Custom(name) => {
                        // Use the registry to get the custom distance function
                        if let Some(distance_func) = get_distance_function(name) {
                            distance_func.distance(vec, query)
                        } else {
                            // Fall back to Euclidean distance if custom function not found
                            ((vec_sq + query_sq_norm - 2.0 * dot(vec, query)).max(0.0)).sqrt()
                        }
                    }
                }
            };
            (*id, dist)
        })
        .collect();

    // Cap k to prevent out-of-bounds panic
    let k = k.min(results.len());
    if k == 0 {
        return (vec![], vec![]);
    }

    let count = results.len().min(k);
    if count > 0 {
        results.select_nth_unstable_by(count - 1, |a, b| a.1.total_cmp(&b.1));
    }
    results.truncate(count);


    let ids = results.iter().map(|(i, _)| *i).collect();
    let dists = results.iter().map(|(_, d)| *d).collect();
    (ids, dists)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: input slices must have the same length");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

