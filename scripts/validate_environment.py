from __future__ import annotations

import os, platform, sys

print("Python", sys.version)
print("Platform", platform.platform())
print("HF_TOKEN", "set" if os.environ.get("HF_TOKEN") else "not set")
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed", exc)
