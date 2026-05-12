

#include <metal_stdlib>
using namespace metal;

inline float gelu_tanh(float x) {

    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + precise::tanh(inner));
}

kernel void fused_ffn(
    device const float* x       [[buffer(0)]],
    device const float* W_up    [[buffer(1)]],
    device const float* b_up    [[buffer(2)]],
    device const float* W_down  [[buffer(3)]],
    device const float* b_down  [[buffer(4)]],
    device float*       out     [[buffer(5)]],
    constant uint&      D_in    [[buffer(6)]],
    constant uint&      D_mid   [[buffer(7)]],
    constant uint&      D_out   [[buffer(8)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  tg_size   [[threads_per_threadgroup]]
)
{

    const uint row = tg_id;

    device const float* x_row = x + row * D_in;

    threadgroup float intermediate[5120];

    for (uint j = tid; j < D_mid; j += tg_size) {

        float acc = b_up[j];
        for (uint k = 0; k < D_in; k++) {
            acc += x_row[k] * W_up[k * D_mid + j];
        }
        intermediate[j] = gelu_tanh(acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + row * D_out;

    for (uint j = tid; j < D_out; j += tg_size) {
        float acc = b_down[j];
        for (uint k = 0; k < D_mid; k++) {
            acc += intermediate[k] * W_down[k * D_out + j];
        }
        out_row[j] = acc;
    }
}

kernel void fused_ln_ffn(
    device const float* x       [[buffer(0)]],
    device const float* ln_w    [[buffer(1)]],
    device const float* ln_b    [[buffer(2)]],
    device const float* W_up    [[buffer(3)]],
    device const float* b_up    [[buffer(4)]],
    device const float* W_down  [[buffer(5)]],
    device const float* b_down  [[buffer(6)]],
    device float*       out     [[buffer(7)]],
    constant uint&      D_in    [[buffer(8)]],
    constant uint&      D_mid   [[buffer(9)]],
    constant uint&      D_out   [[buffer(10)]],
    constant float&     ln_eps  [[buffer(11)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  tg_size   [[threads_per_threadgroup]]
)
{
    const uint row = tg_id;
    device const float* x_row = x + row * D_in;

    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];
    threadgroup float ln_result[1280];

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (uint k = tid; k < D_in; k += tg_size) {
        float val = x_row[k];
        local_sum += val;
        local_sq_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(D_in);
    float var = shared_sq_sum[0] / float(D_in) - mean * mean;
    float inv_std = rsqrt(var + ln_eps);

    for (uint k = tid; k < D_in; k += tg_size) {
        ln_result[k] = (x_row[k] - mean) * inv_std * ln_w[k] + ln_b[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float intermediate[5120];

    for (uint j = tid; j < D_mid; j += tg_size) {
        float acc = b_up[j];
        for (uint k = 0; k < D_in; k++) {
            acc += ln_result[k] * W_up[k * D_mid + j];
        }
        intermediate[j] = gelu_tanh(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_row = out + row * D_out;
    for (uint j = tid; j < D_out; j += tg_size) {
        float acc = b_down[j];
        for (uint k = 0; k < D_mid; k++) {
            acc += intermediate[k] * W_down[k * D_out + j];
        }
        out_row[j] = acc;
    }
}
