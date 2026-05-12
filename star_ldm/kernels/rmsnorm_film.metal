

#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_film_kernel(
    device const float* x           [[buffer(0)]],
    device const float* gamma       [[buffer(1)]],
    device const float* film_scale  [[buffer(2)]],
    device const float* film_shift  [[buffer(3)]],
    device float*       out         [[buffer(4)]],
    constant uint&      D           [[buffer(5)]],
    constant float&     dim_scale   [[buffer(6)]],
    constant uint&      film_stride [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {

    uint row = tgid;
    uint base = row * D;

    float partial_sum = 0.0f;
    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        partial_sum += val * val;
    }
    shared[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm_val = max(sqrt(shared[0]), 1e-8f);

    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        float normalized = val / norm_val * dim_scale;
        float scaled = normalized * gamma[d];
        out[base + d] = scaled * (film_scale[base + d] + 1.0f) + film_shift[base + d];
    }
}

kernel void rmsnorm_kernel(
    device const float* x           [[buffer(0)]],
    device const float* gamma       [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      D           [[buffer(3)]],
    constant float&     dim_scale   [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {
    uint base = tgid * D;

    float partial_sum = 0.0f;
    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        partial_sum += val * val;
    }
    shared[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm_val = max(sqrt(shared[0]), 1e-8f);

    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        out[base + d] = val / norm_val * dim_scale * gamma[d];
    }
}
