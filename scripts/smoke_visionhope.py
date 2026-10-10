"""Build VisionHOPE CUDA extensions and run a forward/backward smoke test on one bag."""
import sys
import time

import torch

from visionhope.models.factory import create_model
from visionhope.models.srnl import SRNL


def main(n_slices=60, size=224, ckpt="weights/visionhope_tiny.pth"):
    torch.manual_seed(0)
    model = create_model("visionhope_tiny", checkpoint_path=ckpt).cuda()
    model.head = torch.nn.Identity()
    model.set_grad_checkpointing(True)
    model.train()
    x = torch.randn(n_slices, 3, size, size, device="cuda")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = model.forward_features(x)
    loss = feats.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    print(f"forward_features shape={tuple(feats.shape)} dtype={feats.dtype} "
          f"fwd+bwd {time.time() - t0:.1f}s peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"

    srnl = SRNL(dim=256, head_dim=16).cuda()
    seq = torch.randn(2, n_slices, 256, device="cuda", requires_grad=True)
    out = srnl(seq)
    out.sum().backward()
    print(f"SRNL standalone out={tuple(out.shape)} grad_ok={torch.isfinite(seq.grad).all().item()}")

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t0 = time.time()
        for _ in range(3):
            model.forward_features(x)
        torch.cuda.synchronize()
    print(f"eval throughput {3 * n_slices / (time.time() - t0):.0f} slices/s")
    print("VISIONHOPE_OK")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
