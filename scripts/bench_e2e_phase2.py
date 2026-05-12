import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

def make_model(device):
    cfg = OmegaConf.create({
        'dataset_name': 'fineweb_100b',
        'train': {'freeze_gpt': True, 'lm_name': 'gpt2-large', 'global_norm': True},
        'sampling': {'noise_schedule_name': 'cosine', 'noise_schedule_scale': 1.0},
        'diffusion_loss': {'weighting_name': 'sigmoid',
                           'weighting_kwargs': {'gamma_shift': 0.0},
                           'train_schedule': 'cosine', 'cosine_shift': 0.0},
        'prompt_generator': {'dim': 1024, 'dim_head': 64, 'depth': 6,
                             'prompt_length': 8, 'dropout': 0.0},
        'scorenet_head': {'dim': 1024, 'depth': 6, 'dropout': 0.0,
                          'output_dim_mult': 4},
    })
    from star_ldm.models.transfusion import TransfusionGPT
    return TransfusionGPT(
        dataset_name='fineweb_100b', transfusion_cfg=cfg,
        gpt2_model_name='gpt2-large', gamma_min=-15, gamma_max=15,
        clf_guidance_dropout=0.1, scale_by_std=True, global_norm=True,
    ).to(device).eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', type=str, default='64,256,512')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--out', type=str, default='results/phase2_prefix_scaling.json')
    args = parser.parse_args()

    dev = torch.device('mps')
    print(f'Building model on {dev}...')
    model = make_model(dev)
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks
    swap_to_fused_blocks(model.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model.score_net_head.transformer)

    gen_kwargs = {'do_sample': True, 'max_new_tokens': 8,
                  'pad_token_id': model.tokenizer.eos_token_id,
                  'top_p': 0.9, 'repetition_penalty': 1.2}

    def run(input_ids):
        with torch.no_grad():
            return model.sample_with_kv_cache(
                input_ids, sampling_timesteps=args.steps, generate_kwargs=gen_kwargs)

    def bench(label, input_ids):
        for _ in range(args.warmup):
            run(input_ids)
        times = []
        for _ in range(args.runs):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            run(input_ids)
            torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f'  {label:<40} mean={arr.mean():7.1f} ms  std={arr.std():5.1f}')
        return {'mean': float(arr.mean()), 'std': float(arr.std()),
                'median': float(np.median(arr)), 'times': arr.tolist()}

    configs = [
        ('A_jit_only',           {'STAR_DISABLE_METAL': '1', 'STAR_DISABLE_V_DDPM_FUSED': '1'}),
        ('B_metal_kernels',      {'STAR_DISABLE_METAL': '0', 'STAR_DISABLE_V_DDPM_FUSED': '1'}),
        ('C_metal_plus_v_ddpm',  {'STAR_DISABLE_METAL': '0', 'STAR_DISABLE_V_DDPM_FUSED': '0'}),
    ]

    results = {}
    for prefix_len in [int(p) for p in args.prefixes.split(',')]:
        input_ids = torch.randint(0, 50257, (1, prefix_len), device=dev)
        print(f'\n=== prefix_len={prefix_len}, steps={args.steps}, runs={args.runs} ===')
        results[str(prefix_len)] = {}
        for cname, env in configs:
            for k, v in env.items():
                os.environ[k] = v

            from star_ldm.diffusion import fused_ops
            from star_ldm.models.modules import fused_blocks
            fused_ops._metal_ddpm_checked = False
            fused_ops._metal_v_ddpm_checked = False
            fused_blocks._metal_rmsnorm_checked = False
            fused_blocks._metal_attn_checked = False
            results[str(prefix_len)][cname] = bench(cname, input_ids)

    print('\n=== Summary (speedup vs JIT-only baseline A) ===')
    for prefix_len, conf_results in results.items():
        print(f'  prefix={prefix_len}:')
        baseline = conf_results['A_jit_only']['mean']
        for cname, stats in conf_results.items():
            print(f'    {cname:<30} {stats["mean"]:7.1f} ms  ({baseline / stats["mean"]:.3f}x)')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {args.out}')

if __name__ == '__main__':
    main()
