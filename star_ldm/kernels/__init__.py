import os
import logging
import torch

logger = logging.getLogger(__name__)

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))

_KERNEL_SPECS = {
    'ddpm_step':      ('metal_ddpm_step',      'ddpm_step.mm'),
    'rmsnorm_film':   ('metal_rmsnorm_film',   'rmsnorm_film.mm'),
    'tiny_attention': ('metal_tiny_attention', 'tiny_attention.mm'),
    'spec_verify':    ('metal_spec_verify',    'spec_verify.mm'),
    'decode_attn':    ('metal_decode_attn',    'decode_attention.mm'),
    'fused_ffn':      ('metal_fused_ffn',      'fused_ffn.mm'),
    'fused_v_ddpm':   ('metal_fused_v_ddpm',   'fused_v_ddpm.mm'),
}

_loaded = {}

def _mps_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def _load(key):
    if key in _loaded:
        return _loaded[key]
    if not _mps_available():
        _loaded[key] = None
        return None
    name, src = _KERNEL_SPECS[key]
    try:
        from torch.utils.cpp_extension import load
        mod = load(
            name=name,
            sources=[os.path.join(_KERNEL_DIR, src)],
            extra_cflags=['-std=c++17', '-ObjC++', '-O2'],
            extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation'],
            verbose=False,
        )
    except Exception as e:
        logger.warning("Metal kernel '%s' failed to load: %s", name, e)
        mod = None
    _loaded[key] = mod
    return mod

def get_ddpm_step_kernel():       return _load('ddpm_step')
def get_rmsnorm_film_kernel():    return _load('rmsnorm_film')
def get_tiny_attention_kernel():  return _load('tiny_attention')
def get_spec_verify_kernel():     return _load('spec_verify')
def get_decode_attention_kernel():return _load('decode_attn')
def get_fused_ffn_kernel():       return _load('fused_ffn')
def get_fused_v_ddpm_kernel():    return _load('fused_v_ddpm')

def preload_all_kernels():
    for key in _KERNEL_SPECS:
        _load(key)
