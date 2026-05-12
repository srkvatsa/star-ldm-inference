

#include <metal_stdlib>
using namespace metal;

kernel void ddpm_step_kernel(
    device const float* z_t        [[buffer(0)]],
    device const float* eps        [[buffer(1)]],
    device const float* noise      [[buffer(2)]],
    device const float* alpha2     [[buffer(3)]],
    device const float* alpha2_next [[buffer(4)]],
    constant float&     vl         [[buffer(5)]],
    device float*       z_out      [[buffer(6)]],
    constant uint&      D          [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint b = tid / D;
    float a2      = alpha2[b];
    float a2_next = alpha2_next[b];
    float a2_now = a2 / a2_next;

    float min_var = exp(log(max(1.0f - a2_next, 1e-8f)) - log(max(1.0f - a2, 1e-8f))) * (1.0f - a2_now);
    float max_var = 1.0f - a2_now;
    min_var = max(min_var, 1e-8f);
    max_var = max(max_var, 1e-8f);
    float sigma = exp(vl * log(max_var) + (1.0f - vl) * log(min_var));

    float inv_sqrt_a2_now = rsqrt(a2_now);
    float coeff = (1.0f - a2_now) * rsqrt(1.0f - a2);

    z_out[tid] = inv_sqrt_a2_now * (z_t[tid] - coeff * eps[tid]) + sqrt(sigma) * noise[tid];
}
