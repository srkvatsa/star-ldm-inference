# Contribution Map

This repository adds an inference-optimization layer for Apple Silicon on top of [STAR-LDM](https://github.com/justinlovelace/STAR-LDM) (Lovelace et al., COLM 2025). The model architecture and training code come from the upstream project; everything related to Metal kernels, KV-cache amortization, the streamlined GPT-2 forward, the closed-form fused v+DDPM step, and the benchmark and measurement code is this project's work.

## Files authored by this project

| Area | Files |
|------|-------|
| Production Metal kernels | `star_ldm/kernels/{ddpm_step, fused_v_ddpm, rmsnorm_film, tiny_attention}.{metal,mm}` |
| Research-artifact Metal kernels (documented negative or shelved results) | `star_ldm/kernels/{decode_attention, fused_ffn, spec_verify}.{metal,mm}` |
| Kernel loader | `star_ldm/kernels/__init__.py` |
| Python integration | `star_ldm/diffusion/fused_ops.py`, `star_ldm/models/modules/fused_blocks.py`, `star_ldm/models/decode_attention_patch.py`, `star_ldm/diffusion/async_schedule.py` |
| Speculative decoding (implementation complete, draft model never trained) | `star_ldm/decoding/{__init__.py, speculative.py, draft_model.py}` |
| Benchmarks and measurement scripts | `scripts/bench_e2e_phase2.py`, `scripts/eval_c4_canonical.py`, `scripts/fusion_crossover.py`, `scripts/profile_inference.py`, `scripts/roofline_analysis.py`, `scripts/benchmark_metal.py`, `scripts/validate_kv_cache.py` |
| Figure and poster generation | `scripts/plot_*.py`, `scripts/poster/*` |
| Older or one-off experimental scripts | `scripts/archive/*` |
| Correctness tests | `tests/test_metal_kernels.py`, `tests/test_speculative.py` |
| Planning and poster docs | `docs/*` |
| Benchmark outputs and figures | `results/*.json`, `figures/*.png`, `figures/*.pdf` |

## Files modified from upstream STAR-LDM

| File | Notes |
|------|-------|
| `star_ldm/models/transfusion.py` | Added `_fast_gpt2_forward`, `sample_with_kv_cache`, `_compute_prefix_kv_cache`, `_prepare_kv_buffers`, `_v_pred_cached`, `_diffusion_model_predictions_cached`, `_fast_generate`, plus the speculative and Picard samplers (research-only). |
| `star_ldm/interface.py` | KV-cache option and speculative entry points on `TransfusionGPTInterface`. |
| `star_ldm/diffusion/diff_utils.py` | Utility additions. |
| `star_ldm/diffusion/noise_schedule.py` | Small fix. |
| `star_ldm/models/classifier.py` | Minor edits. |
| `README.md` | Rewritten for this project. |
| `.gitignore` | Added entries for checkpoints and build artifacts. |

## Files unchanged from upstream

Everything not listed above (the model architecture under `star_ldm/models/modules/`, the diffusion utilities, the tokenizer pieces, the dataset and training utilities) is the original STAR-LDM implementation by Lovelace et al. The full upstream codebase lives at https://github.com/justinlovelace/STAR-LDM.

## Code map

| Component | Where it lives |
|-----------|----------------|
| Streamlined GPT-2 forward + KV-cache reuse | `star_ldm/models/transfusion.py:sample_with_kv_cache`, `:_fast_gpt2_forward`, `:_prepare_kv_buffers`, `:_compute_prefix_kv_cache` |
| Fused RMSNorm + FiLM | `star_ldm/kernels/rmsnorm_film.{metal,mm}`, swapped in via `swap_to_fused_blocks` in `fused_blocks.py` |
| Tiny (S=8) fused attention | `star_ldm/kernels/tiny_attention.{metal,mm}`, used by `FusedAttention` in `fused_blocks.py` |
| Fused DDPM step (single dispatch) | `star_ldm/kernels/ddpm_step.{metal,mm}` |
| Closed-form fused v-prediction + DDPM | `star_ldm/kernels/fused_v_ddpm.{metal,mm}`; JIT fallback at `star_ldm/diffusion/fused_ops.py:jit_fused_v_ddpm_step` |
| Wrapper-overhead fix (`setBytes` for scalars) | Inline comment in `star_ldm/kernels/ddpm_step.mm` and `fused_v_ddpm.mm`; before/after numbers in `results/kernel_microbench.json` |
| End-to-end and prefix-scaling benchmarks | `scripts/bench_e2e_phase2.py`, `scripts/eval_c4_canonical.py`; outputs in `results/phase2_prefix_scaling.json`, `results/c4_canonical.json` |
