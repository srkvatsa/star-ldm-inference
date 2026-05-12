import torch, time
from torch import Tensor
from star_ldm.diffusion.fused_ops import fused_v_to_x0_eps, _jit_fused_ddpm_step

torch.manual_seed(0)

@torch.jit.script
def jit_fused_v_ddpm(
    z_t: Tensor, v: Tensor, noise: Tensor,
    alpha2: Tensor, alpha2_next: Tensor, var_lambda: float
) -> Tensor:
    sqrt_a2 = torch.sqrt(alpha2)
    sqrt_1ma2 = torch.sqrt(1.0 - alpha2)
    a2_now = alpha2 / alpha2_next
    min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - a2_now)
    max_var = 1.0 - a2_now
    sigma = torch.exp(var_lambda * torch.log(max_var) + (1.0 - var_lambda) * torch.log(min_var))
    inv_sqrt_a2_now = 1.0 / torch.sqrt(a2_now)
    coeff = (1.0 - a2_now) / sqrt_1ma2
    A = inv_sqrt_a2_now * (1.0 - coeff * sqrt_1ma2)
    B = -inv_sqrt_a2_now * coeff * sqrt_a2
    C = torch.sqrt(sigma)
    return A * z_t + B * v + C * noise

@torch.jit.script
def jit_fused_v_x_start(z_t: Tensor, v: Tensor, alpha2: Tensor) -> Tensor:
    return torch.sqrt(alpha2) * z_t - torch.sqrt(1.0 - alpha2) * v

def main():
    dev = torch.device('mps')
    B, D = 1, 768
    z_t = torch.randn(B, D, device=dev)
    v = torch.randn(B, D, device=dev)
    noise = torch.randn(B, D, device=dev)
    alpha2 = torch.tensor([[0.4]], device=dev)
    alpha2_next = torch.tensor([[0.6]], device=dev)
    var_lambda = 0.2

    x_, e = fused_v_to_x0_eps(z_t, v, alpha2)
    ref = _jit_fused_ddpm_step(z_t, e, noise, alpha2, alpha2_next, var_lambda)
    new = jit_fused_v_ddpm(z_t, v, noise, alpha2, alpha2_next, var_lambda)
    torch.mps.synchronize()
    print(f'correctness max diff: {(ref - new).abs().max().item():.2e}')

    def bench(fn, w=200, it=3000):
        for _ in range(w):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter() - t0) / it * 1e6

    old = bench(lambda: _jit_fused_ddpm_step(
        z_t, fused_v_to_x0_eps(z_t, v, alpha2)[1],
        noise, alpha2, alpha2_next, var_lambda
    ))
    new_us = bench(lambda: jit_fused_v_ddpm(z_t, v, noise, alpha2, alpha2_next, var_lambda))
    print(f'old (v_to_x0_eps + jit_ddpm):  {old:7.2f} us')
    print(f'new (closed-form jit):         {new_us:7.2f} us')
    print(f'speedup: {old/new_us:.2f}x')

if __name__ == '__main__':
    main()
