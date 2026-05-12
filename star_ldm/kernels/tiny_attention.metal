

#include <metal_stdlib>
using namespace metal;

constant uint SEQ_LEN = 8;

kernel void tiny_attention_kernel(
    device const float* Q           [[buffer(0)]],
    device const float* K           [[buffer(1)]],
    device const float* V           [[buffer(2)]],
    device const float* q_gamma     [[buffer(3)]],
    device const float* k_gamma     [[buffer(4)]],
    device float*       out         [[buffer(5)]],
    constant uint&      D_head      [[buffer(6)]],
    constant float&     dim_head_scale [[buffer(7)]],
    constant float&     attn_scale  [[buffer(8)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {

    uint bh_offset = tgid * SEQ_LEN * D_head;

    threadgroup float* Q_norm = shared;
    threadgroup float* K_norm = shared + SEQ_LEN * D_head;
    threadgroup float* scratch = shared + 2 * SEQ_LEN * D_head;

    for (uint s = 0; s < SEQ_LEN; s++) {
        uint vec_offset = bh_offset + s * D_head;

        float partial = 0.0f;
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = Q[vec_offset + d];
            partial += val * val;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float q_norm_val = max(sqrt(scratch[0]), 1e-8f);

        for (uint d = tid; d < D_head; d += tg_sz) {
            Q_norm[s * D_head + d] = Q[vec_offset + d] / q_norm_val * dim_head_scale * q_gamma[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint s = 0; s < SEQ_LEN; s++) {
        uint vec_offset = bh_offset + s * D_head;

        float partial = 0.0f;
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = K[vec_offset + d];
            partial += val * val;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float k_norm_val = max(sqrt(scratch[0]), 1e-8f);

        for (uint d = tid; d < D_head; d += tg_sz) {
            K_norm[s * D_head + d] = K[vec_offset + d] / k_norm_val * dim_head_scale * k_gamma[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* attn_matrix = scratch;

    for (uint idx = tid; idx < SEQ_LEN * SEQ_LEN; idx += tg_sz) {
        uint i = idx / SEQ_LEN;
        uint j = idx % SEQ_LEN;
        float dot = 0.0f;
        for (uint d = 0; d < D_head; d++) {
            dot += Q_norm[i * D_head + d] * K_norm[j * D_head + d];
        }
        attn_matrix[idx] = dot * attn_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < SEQ_LEN; i += tg_sz) {

        float max_val = attn_matrix[i * SEQ_LEN];
        for (uint j = 1; j < SEQ_LEN; j++) {
            max_val = max(max_val, attn_matrix[i * SEQ_LEN + j]);
        }

        float sum_exp = 0.0f;
        for (uint j = 0; j < SEQ_LEN; j++) {
            float e = exp(attn_matrix[i * SEQ_LEN + j] - max_val);
            attn_matrix[i * SEQ_LEN + j] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (uint j = 0; j < SEQ_LEN; j++) {
            attn_matrix[i * SEQ_LEN + j] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < SEQ_LEN; i++) {
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = 0.0f;
            for (uint j = 0; j < SEQ_LEN; j++) {
                val += attn_matrix[i * SEQ_LEN + j] * V[bh_offset + j * D_head + d];
            }
            out[bh_offset + i * D_head + d] = val;
        }
    }
}
