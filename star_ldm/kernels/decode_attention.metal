

#include <metal_stdlib>
using namespace metal;

inline float simd_reduce_add(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

constant constexpr uint D64 = 64;
constant constexpr uint VEC_SIZE = 4;
constant constexpr uint D_VECS = D64 / VEC_SIZE;
constant constexpr uint TPQ = 32;

kernel void decode_n_attention(
    device const float* Q          [[buffer(0)]],
    device const float* K          [[buffer(1)]],
    device const float* V          [[buffer(2)]],
    device float*       O          [[buffer(3)]],
    constant uint&      n_q        [[buffer(4)]],
    constant uint&      s_kv       [[buffer(5)]],
    constant uint&      d_head     [[buffer(6)]],
    constant uint&      n_heads    [[buffer(7)]],
    constant float&     scale      [[buffer(8)]],
    uint  tg_id        [[threadgroup_position_in_grid]],
    uint  tid_in_tg    [[thread_index_in_threadgroup]]
)
{
    const uint batch_idx = tg_id / n_heads;
    const uint head_idx  = tg_id % n_heads;
    const uint q_idx = tid_in_tg / TPQ;
    const uint lane  = tid_in_tg % TPQ;

    if (q_idx >= n_q) return;

    const uint bh = batch_idx * n_heads + head_idx;
    device const float* Q_ptr = Q + bh * n_q * d_head;
    device const float* K_ptr = K + bh * s_kv * d_head;
    device const float* V_ptr = V + bh * s_kv * d_head;
    device float*       O_ptr = O + bh * n_q * d_head;

    const uint d0 = lane * 2;
    const uint d1 = lane * 2 + 1;

    float q0 = (d0 < d_head) ? Q_ptr[q_idx * d_head + d0] : 0.0f;
    float q1 = (d1 < d_head) ? Q_ptr[q_idx * d_head + d1] : 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    float o0 = 0.0f;
    float o1 = 0.0f;

    for (uint s = 0; s < s_kv; s++) {

        float k0 = (d0 < d_head) ? K_ptr[s * d_head + d0] : 0.0f;
        float k1 = (d1 < d_head) ? K_ptr[s * d_head + d1] : 0.0f;
        float partial = q0 * k0 + q1 * k1;

        float score = simd_reduce_add(partial) * scale;

        float prev_max = row_max;
        row_max = max(row_max, score);
        float exp_diff = exp(prev_max - row_max);
        float exp_score = exp(score - row_max);

        row_sum = row_sum * exp_diff + exp_score;

        o0 *= exp_diff;
        o1 *= exp_diff;

        float v0 = (d0 < d_head) ? V_ptr[s * d_head + d0] : 0.0f;
        float v1 = (d1 < d_head) ? V_ptr[s * d_head + d1] : 0.0f;
        o0 += exp_score * v0;
        o1 += exp_score * v1;
    }

    float inv_sum = 1.0f / row_sum;
    if (d0 < d_head) O_ptr[q_idx * d_head + d0] = o0 * inv_sum;
    if (d1 < d_head) O_ptr[q_idx * d_head + d1] = o1 * inv_sum;
}

kernel void decode_n_attention_f16(
    device const half*  Q          [[buffer(0)]],
    device const half*  K          [[buffer(1)]],
    device const half*  V          [[buffer(2)]],
    device half*        O          [[buffer(3)]],
    constant uint&      n_q        [[buffer(4)]],
    constant uint&      s_kv       [[buffer(5)]],
    constant uint&      d_head     [[buffer(6)]],
    constant uint&      n_heads    [[buffer(7)]],
    constant float&     scale      [[buffer(8)]],
    uint  tg_id        [[threadgroup_position_in_grid]],
    uint  tid_in_tg    [[thread_index_in_threadgroup]]
)
{
    const uint batch_idx = tg_id / n_heads;
    const uint head_idx  = tg_id % n_heads;
    const uint q_idx = tid_in_tg / TPQ;
    const uint lane  = tid_in_tg % TPQ;

    if (q_idx >= n_q) return;

    const uint bh = batch_idx * n_heads + head_idx;
    device const half* Q_ptr = Q + bh * n_q * d_head;
    device const half* K_ptr = K + bh * s_kv * d_head;
    device const half* V_ptr = V + bh * s_kv * d_head;
    device half*       O_ptr = O + bh * n_q * d_head;

    const uint d0 = lane * 2;
    const uint d1 = lane * 2 + 1;

    float q0 = (d0 < d_head) ? float(Q_ptr[q_idx * d_head + d0]) : 0.0f;
    float q1 = (d1 < d_head) ? float(Q_ptr[q_idx * d_head + d1]) : 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;

    for (uint s = 0; s < s_kv; s++) {
        float k0 = (d0 < d_head) ? float(K_ptr[s * d_head + d0]) : 0.0f;
        float k1 = (d1 < d_head) ? float(K_ptr[s * d_head + d1]) : 0.0f;
        float partial = q0 * k0 + q1 * k1;

        float score = simd_reduce_add(partial) * scale;

        float prev_max = row_max;
        row_max = max(row_max, score);
        float exp_diff = exp(prev_max - row_max);
        float exp_score = exp(score - row_max);

        row_sum = row_sum * exp_diff + exp_score;
        o0 *= exp_diff;
        o1 *= exp_diff;

        float v0 = (d0 < d_head) ? float(V_ptr[s * d_head + d0]) : 0.0f;
        float v1 = (d1 < d_head) ? float(V_ptr[s * d_head + d1]) : 0.0f;
        o0 += exp_score * v0;
        o1 += exp_score * v1;
    }

    float inv_sum = 1.0f / row_sum;
    if (d0 < d_head) O_ptr[q_idx * d_head + d0] = half(o0 * inv_sum);
    if (d1 < d_head) O_ptr[q_idx * d_head + d1] = half(o1 * inv_sum);
}
