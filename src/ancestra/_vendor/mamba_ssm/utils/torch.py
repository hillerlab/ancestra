# Vendored from mamba_ssm==2.3.2.post1; source SHA-256: 1c3132a1a747e914874e84b5f300ea80d0297faba6eb9256cb5ac4b3d6413c06; imports: mamba_ssm.* -> ancestra._vendor.mamba_ssm.*.
import torch
from functools import partial
from typing import Callable

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
