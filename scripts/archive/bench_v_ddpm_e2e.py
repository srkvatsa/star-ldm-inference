import argparse, time
import torch
from omegaconf import OmegaConf

def make_model(device):
    cfg = OmegaConf.create({
        'dataset_name': 'fineweb_100b',
        'train': {'freeze_gpt': True, 'lm_name': 'gpt2-large', 'global_norm': True},
        'sampling': {'noise_schedule_name': 'cosine', 'noise_schedule_scale': 1.0},
        'diffusion_loss': {
            'weighting_name': 'sigmoid',
            'weighting_kwargs': {'gamma_shift': 0.0},
            'train_schedule': 'cosine',
            'cosine_shift': 0.0,
        },
        'prompt_generator': {'dim': 1024, 'dim_head': 64, 'depth': 6, 'prompt_length': 8, 'dropout': 0.0},
        'scorenet_head': {'dim': 1024, 'depth': 6, 'dropout': 0.0, 'output_dim_mult': 4},
    })
    from star_ldm.models.transfusion import TransfusionGPT
    model = TransfusionGPT(
        dataset_name='fineweb_100b',
        transfusion_cfg=cfg,
        gpt2_model_name='gpt2-large',
        gamma_min=-15, gamma_max=15,
        clf_guidance_dropout=0.1,
        scale_by_std=True,
        global_norm=True,
    )
    model = model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_len', type=int, default=64)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()

    dev = torch.device('mps')
    print(f'Loading dummy model on {dev}...')
    model = make_model(dev)

    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks
    swap_to_fused_blocks(model.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model.score_net_head.transformer)

    input_ids = torch.randint(0, 50257, (1, args.prefix_len), device=dev)

    gen_kwargs = {"do_sample": True, "max_new_tokens": 8,
                  "pad_token_id": model.tokenizer.eos_token_id,
                  "top_p": 0.9, "repetition_penalty": 1.2}

    def run_old():

        with torch.no_grad():
            return model.sample_with_kv_cache(
                input_ids, sampling_timesteps=args.steps,
                generate_kwargs=gen_kwargs)

    print(f'Warmup ({args.warmup} runs)...')
    for _ in range(args.warmup):
        with torch.no_grad():
            run_old()

    times = []
    for i in range(args.runs):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            run_old()
        torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        print(f'  run {i+1}: {dt:.1f} ms')

    import numpy as np
    arr = np.array(times)
    print(f'\nMean: {arr.mean():.1f} ms  std: {arr.std():.1f}  median: {np.median(arr):.1f}')

if __name__ == '__main__':
    main()
