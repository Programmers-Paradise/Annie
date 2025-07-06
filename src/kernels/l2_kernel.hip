#include <hip/hip_runtime.h>

extern "C" __global__ void l2_distance_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ corpus,
    float* __restrict__ out,
    int n_queries,
    int n_vectors,
    int dim
) {
    int q_idx = blockIdx.x;
    int v_idx = threadIdx.x;

    if (q_idx < n_queries && v_idx < n_vectors) {
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = queries[q_idx * dim + d] - corpus[v_idx * dim + d];
            sum += diff * diff;
        }
        out[q_idx * n_vectors + v_idx] = sum;
    }
}