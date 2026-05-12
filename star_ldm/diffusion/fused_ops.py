import os
import torch
from torch import Tensor

_metal_ddpm = None
_metal_v_ddpm = None
_metal_ddpm_checked = False
_metal_v_ddpm_checked = False

def _metal_disabled():
    return os.environ.get('STAR_DISABLE_METAL') == '1'

def _get_metal_ddpm():
    global _metal_ddpm, _metal_ddpm_checked
    if _metal_disabled():
        return None
    if not _metal_ddpm_checked:
        _metal_ddpm_checked = True
        from star_ldm.kernels import get_ddpm_step_kernel
        _metal_ddpm = get_ddpm_step_kernel()
    return _metal_ddpm

def _get_metal_v_ddpm():
    global _metal_v_ddpm, _metal_v_ddpm_checked
    if _metal_disabled():
        return None
    if not _metal_v_ddpm_checked:
        _metal_v_ddpm_checked = True
        from star_ldm.kernels import get_fused_v_ddpm_kernel
        _metal_v_ddpm = get_fused_v_ddpm_kernel()
    return _metal_v_ddpm

def metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda):
    mod = _get_metal_ddpm()
    if mod is None or not z_t.is_mps:
        return None
    return mod.ddpm_step(
        z_t.contiguous(), eps.contiguous(), noise.contiguous(),
        alpha2.contiguous(), alpha2_next.contiguous(), float(var_lambda),
    )

def metal_fused_v_ddpm(z_t, v, noise, alpha2, alpha2_next, var_lambda, is_last_step=False):
    mod = _get_metal_v_ddpm()
    if mod is None or not z_t.is_mps:
        return None
    return mod.fused_v_ddpm(
        z_t.contiguous(), v.contiguous(), noise.contiguous(),
        alpha2.contiguous(), alpha2_next.contiguous(),
        float(var_lambda), bool(is_last_step),
    )

@torch.jit.script
def _jit_fused_ddpm_step(
    z_t: Tensor, eps: Tensor, noise: Tensor,
    alpha2: Tensor, alpha2_next: Tensor, var_lambda: float,
) -> Tensor:
    alpha2_now = alpha2 / alpha2_next
    min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
    max_var = 1.0 - alpha2_now
    sigma = torch.exp(var_lambda * torch.log(max_var) + (1.0 - var_lambda) * torch.log(min_var))
    inv_sqrt_alpha2_now = 1.0 / torch.sqrt(alpha2_now)
    coeff = (1.0 - alpha2_now) / torch.sqrt(1.0 - alpha2)
    return inv_sqrt_alpha2_now * (z_t - coeff * eps) + torch.sqrt(sigma) * noise

@torch.jit.script
def fused_ddim_step(x_start: Tensor, eps: Tensor, alpha2_next: Tensor) -> Tensor:
    return torch.sqrt(alpha2_next) * x_start + torch.sqrt(1.0 - alpha2_next) * eps

@torch.jit.script
def jit_fused_v_ddpm_step(
    z_t: Tensor, v: Tensor, noise: Tensor,
    alpha2: Tensor, alpha2_next: Tensor, var_lambda: float,
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

@torch.jit.script
def fused_v_to_x0_eps(z_t: Tensor, v: Tensor, alpha2: Tensor) -> tuple[Tensor, Tensor]:
    sqrt_alpha2 = torch.sqrt(alpha2)
    sqrt_one_minus_alpha2 = torch.sqrt(1.0 - alpha2)
    x_start = sqrt_alpha2 * z_t - sqrt_one_minus_alpha2 * v
    eps = sqrt_alpha2 * v + sqrt_one_minus_alpha2 * z_t
    return x_start, eps

def fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda):
    out = metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)
    if out is not None:
        return out
    return _jit_fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)

def fused_v_ddpm_step(z_t, v, noise, alpha2, alpha2_next, var_lambda, is_last_step=False):
    out = metal_fused_v_ddpm(z_t, v, noise, alpha2, alpha2_next, var_lambda, is_last_step)
    if out is not None:
        return out
    if is_last_step:
        return jit_fused_v_x_start(z_t, v, alpha2)
    return jit_fused_v_ddpm_step(z_t, v, noise, alpha2, alpha2_next, var_lambda)
