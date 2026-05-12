
import argparse
import time
import torch
from omegaconf import OmegaConf

def get_device():
 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
 return torch.device("mps")
 if torch.cuda.is_available():
 return torch.device("cuda")
 return torch.device("cpu")

def sync(device):
 if device.type == "mps":
 torch.mps.synchronize()
 elif device.type == "cuda":
 torch.cuda.synchronize()

def load_model(device, model_path=None):
 if model_path:
 from star_ldm.interface import TransfusionGPTInterface
 interface = TransfusionGPTInterface(model_path=model_path, device=str(device))
 return interface.model
 cfg = OmegaConf.create({
 "dataset_name": "fineweb_100b",
 "train": {"freeze_gpt": True, "lm_name": "gpt2-large", "global_norm": True},
 "sampling": {"noise_schedule_name": "cosine", "noise_schedule_scale": 1.0},
 "diffusion_loss": {
 "weighting_name": "sigmoid",
 "weighting_kwargs": {"gamma_shift": 0.0},
 "train_schedule": "cosine",
 "cosine_shift": 0.0,
 },
 "prompt_generator": {"dim": 1024, "dim_head": 64, "depth": 6, "prompt_length": 8, "dropout": 0.0},
 "scorenet_head": {"dim": 1024, "depth": 6, "dropout": 0.0, "output_dim_mult": 4},
 })
 from star_ldm.models.transfusion import TransfusionGPT
 model = TransfusionGPT(
 dataset_name="fineweb_100b", transfusion_cfg=cfg,
 gpt2_model_name="gpt2-large",
 gamma_min=-15, gamma_max=15,
 clf_guidance_dropout=0.1, scale_by_std=True, global_norm=True,
 )
 return model.to(device).eval()

def phase_gpt2_only(model, device, repeats=10):
 from star_ldm.models.transfusion import _get_lm_dtype

 gpt2 = model.gpt2
 tokenizer = model.tokenizer
 input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to(device)

 with torch.no_grad():
 prefix_out = gpt2(input_ids, use_cache=True)
 kv_cache = prefix_out.past_key_values

 soft_tokens = torch.randn(1, 8, gpt2.config.n_embd, device=device, dtype=torch.float32)

 sync(device)
 print(f" Phase: GPT-2 forward (8 tokens, KV-cached), {repeats} repeats")
 print(f" Look for: 36-layer blocks, each with LayerNorm→c_attn→SDPA→c_proj→LayerNorm→MLP")

 for i in range(repeats):
 with torch.no_grad():
 gpt2(inputs_embeds=soft_tokens, past_key_values=kv_cache,
 use_cache=False, output_hidden_states=True)

 sync(device)

def phase_gpt2_no_cache(model, device, repeats=10):
 from star_ldm.models.transfusion import _get_lm_dtype

 gpt2 = model.gpt2
 tokenizer = model.tokenizer
 input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to(device)

 prefix_embed = gpt2.transformer.wte(input_ids)
 soft_tokens = torch.randn(1, 8, gpt2.config.n_embd, device=device, dtype=torch.float32)
 full_embed = torch.cat([prefix_embed, soft_tokens], dim=1)

 sync(device)
 print(f" Phase: GPT-2 forward (prefix+8 tokens, no cache), {repeats} repeats")

 for i in range(repeats):
 with torch.no_grad():
 gpt2(inputs_embeds=full_embed, output_hidden_states=True)
 sync(device)

def phase_single_layer(model, device, repeats=30):
 gpt2 = model.gpt2

 tokenizer = model.tokenizer
 input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to(device)

 with torch.no_grad():
 prefix_out = gpt2(input_ids, use_cache=True)
 kv_cache = prefix_out.past_key_values

 hidden = torch.randn(1, 8, gpt2.config.n_embd, device=device, dtype=torch.float32)
 block = gpt2.transformer.h[0]

 sync(device)
 print(f" Phase: Single GPT-2 layer (layer 0), {repeats} repeats")
 print(f" Look for: ln_1 → c_attn → SDPA → c_proj → residual → ln_2 → MLP → residual")

 for i in range(repeats):
 with torch.no_grad():
 block(hidden, past_key_values=kv_cache, use_cache=False)
 sync(device)

def phase_diffusion_step(model, device, repeats=5):
 from star_ldm.models.transfusion import _get_lm_dtype
 from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
 from star_ldm.diffusion.fused_ops import fused_ddpm_step
 from star_ldm.diffusion.diff_utils import predict_start_from_v, predict_noise_from_v
 from einops import rearrange

 tokenizer = model.tokenizer
 input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to(device)
 batch = 1

 num_diff = model.num_diffusion_tokens
 diff_mask = torch.zeros((batch, input_ids.shape[1] + num_diff), dtype=torch.bool, device=device)
 diff_mask[:, -num_diff:] = True
 padded_ids = torch.nn.functional.pad(input_ids, (0, num_diff), value=tokenizer.pad_token_id)

 sample_noise_schedule = get_scaled_noise_schedule("cosine", scale=3.0)
 z_t = torch.randn((batch, 768), device=device)

 t = torch.tensor([[0.5]], device=device)
 t_next = torch.tensor([[0.48]], device=device)
 alpha2 = sample_noise_schedule(t)
 alpha2_next = sample_noise_schedule(t_next)

 sync(device)
 print(f" Phase: Full diffusion step, {repeats} repeats")
 print(f" Look for: 3 clusters per step SPG (small), GPT-2 (big), ScoreNet (medium)")

 for i in range(repeats):
 with torch.no_grad():

 soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)
 sync(device)

 input_embed = model.lm_embedding(padded_ids).float()
 input_embed[diff_mask] = rearrange(soft_prompt, "b l d -> (b l) d")
 gpt2_out = model.gpt2(
 inputs_embeds=input_embed.to(_get_lm_dtype()),
 output_hidden_states=True,
 )
 sync(device)

 diff_hidden = rearrange(
 gpt2_out.hidden_states[-1][diff_mask],
 "(b l) d -> b l d", b=batch, l=num_diff,
 )
 diff_hidden = torch.cat((soft_prompt, diff_hidden), dim=-1)
 v_pred = model.score_net_head(diff_hidden, time_emb)
 sync(device)

 x_start = predict_start_from_v(z_t, v_pred, alpha2)
 eps = predict_noise_from_v(z_t, v_pred, alpha2)
 noise = torch.randn_like(z_t)
 z_t = fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)
 sync(device)

def phase_micro_transformers(model, device, repeats=20):
 from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule

 sample_noise_schedule = get_scaled_noise_schedule("cosine", scale=3.0)
 z_t = torch.randn((1, 768), device=device)
 t = torch.tensor([[0.5]], device=device)
 alpha2 = sample_noise_schedule(t)

 sync(device)
 print(f" Phase: SoftPromptGen, {repeats} repeats")

 for i in range(repeats):
 with torch.no_grad():
 soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)
 sync(device)

 hidden_gpt2 = torch.randn(1, 8, 1280, device=device)
 diff_hidden = torch.cat((soft_prompt, hidden_gpt2), dim=-1)

 sync(device)
 print(f" Phase: ScoreNetHead, {repeats} repeats")

 for i in range(repeats):
 with torch.no_grad():
 model.score_net_head(diff_hidden, time_emb)
 sync(device)

PHASES = {
 "all": None,
 "gpt2_only": phase_gpt2_only,
 "gpt2_no_cache": phase_gpt2_no_cache,
 "single_layer": phase_single_layer,
 "diffusion_step": phase_diffusion_step,
 "micro_transformers": phase_micro_transformers,
}

def main():
 parser = argparse.ArgumentParser(description="Xcode Metal profiling workload")
 parser.add_argument("--model_path", type=str, default=None)
 parser.add_argument("--dummy", action="store_true")
 parser.add_argument("--phase", type=str, default="all", choices=list(PHASES.keys()),
 help="Which phase to run (default: all)")
 parser.add_argument("--repeats", type=int, default=10,
 help="Repeats per phase (more = easier to see in trace)")
 parser.add_argument("--pause", type=float, default=3.0,
 help="Seconds to pause before capture window (attach Instruments here)")
 args = parser.parse_args()

 if not args.dummy and not args.model_path:
 parser.error("Provide --model_path or --dummy")

 device = get_device()
 print(f"Device: {device}")

 print("Loading model...")
 model = load_model(device, args.model_path if not args.dummy else None)
 print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

 tokenizer = model.tokenizer
 input_ids = tokenizer("warmup", return_tensors="pt").input_ids.to(device)

 print("\nWarmup (compiling Metal shaders)...")
 with torch.no_grad():
 model.gpt2(input_ids, use_cache=True)
 z = torch.randn(1, 768, device=device)
 a = torch.tensor([[0.5]], device=device)
 model.soft_prompt_generator(z, a)
 sync(device)
 print("Warmup done.")

 print(f"\n{'='*60}")
 print(f" Attach Xcode Instruments now (Metal System Trace)")
 print(f" PID: {__import__('os').getpid()}")
 print(f" Waiting {args.pause:.0f}s...")
 print(f"{'='*60}")
 time.sleep(args.pause)

 print(f"\n>>> CAPTURE WINDOW START <<<")
 sync(device)

 if args.phase == "all":
 for name, fn in PHASES.items():
 if fn is not None:
 print(f"\n--- {name} ---")
 fn(model, device, repeats=args.repeats)

 sync(device)
 time.sleep(0.1)
 else:
 PHASES[args.phase](model, device, repeats=args.repeats)

 sync(device)
 print(f"\n>>> CAPTURE WINDOW END <<<")
 print("Stop recording in Instruments now.")

if __name__ == "__main__":
 main()
