"""Peak memory / throughput of one end-to-end step vs slices per step and gradient checkpointing."""
import sys
import time

import torch

from models.abmil import BagModel, build_head
from models.encoder import build_encoder
from utils.config import load_config

cfg = load_config(None)
for ckpt in (False, True):
    cfg["encoder"]["grad_checkpointing"] = ckpt
    enc = build_encoder(cfg)
    model = BagModel(enc, build_head(cfg, enc.out_dim)).cuda().train()
    for n in [int(v) for v in sys.argv[1:]] or [22, 44, 88]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.rand(1, n, 3, 224, 224, device="cuda")
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        try:
            for it in range(3):
                if it == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, mask)
                out["logits"].float().sum().backward()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 2
            print(f"ckpt={ckpt} slices={n:4d} peak={torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB "
                  f"{n / dt:6.0f} slices/s", flush=True)
        except torch.OutOfMemoryError:
            print(f"ckpt={ckpt} slices={n:4d} OOM", flush=True)
            break
        model.zero_grad(set_to_none=True)
    del model, enc
