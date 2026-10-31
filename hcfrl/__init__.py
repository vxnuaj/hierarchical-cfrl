
# HCFRL_FP32=1 disables every torch.amp.autocast region (bf16 is unsupported on pre-Ampere GPUs such as the P100/V100),
# so the same code runs in fp32 on those nodes. Imported by every hcfrl entry point.
import os as _os
if _os.environ.get("HCFRL_FP32") == "1":
    import torch as _torch
    _orig_autocast = _torch.amp.autocast

    class _NoAutocast(_orig_autocast):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            super().__init__(device_type, dtype=dtype, enabled=False, cache_enabled=cache_enabled)

    _torch.amp.autocast = _NoAutocast
