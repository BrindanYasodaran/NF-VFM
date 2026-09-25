"""Frozen flow map: the py-meanflow CIFAR-10 checkpoint net_ema1, whose FID of 2.812 is
reproduced by fid_check.py.

It is unconditional, works in pixel space, and predicts average velocity. One step decodes
as x = z - u(z, t=1, h=1), starting from z ~ N(0, I) of shape (B, 3, 32, 32). K steps
repeat z <- z - (t_k - t_{k+1}) * u(z, t_k, h = t_k - t_{k+1}).

Every CIFAR experiment decodes through this module.
"""
import os, sys, pickle
import torch

COMMON = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{COMMON}/py-meanflow/meanflow")
device = "cuda"
D = 3 * 32 * 32
EMA1 = f"{COMMON}/pymf_ema1.pt"

_net = None
def net():
    global _net
    if _net is None:
        from models.model_configs import instantiate_model
        from train_arg_parser import get_args_parser
        args = get_args_parser().parse_args([])
        args.use_edm_aug = True
        if os.path.exists(EMA1):
            model = instantiate_model(args)
            model.net_ema1.load_state_dict(torch.load(EMA1, map_location="cpu"))
            _net = model.net_ema1.to(device).eval()
        else:
            ck = torch.load(f"{COMMON}/pymf_cifar10.pth", map_location="cpu",
                            weights_only=False)
            model = instantiate_model(args)
            model.load_state_dict(ck["model"])
            torch.save(model.net_ema1.state_dict(), EMA1)
            _net = model.net_ema1.to(device).eval()
        for p in _net.parameters():
            p.requires_grad_(False)
    return _net

def decode1_diff(z_flat):
    """Differentiable 1-step decode: (B,3072) -> (B,3,32,32) approx [-1,1]."""
    z = z_flat.view(-1, 3, 32, 32)
    n = net()
    B = z.shape[0]
    t = torch.ones(B, device=device)
    u = n(z, (t, t), aug_cond=None)  # h = t - r = 1
    return z - u

@torch.no_grad()
def decode1(z_flat, bs=512):
    outs = []
    for i in range(0, z_flat.shape[0], bs):
        outs.append(decode1_diff(z_flat[i:i+bs]).float())
    return torch.cat(outs)

def save_grid(x, path, nrow=16):
    from torchvision.utils import make_grid, save_image
    save_image(make_grid((x[:nrow*8].clamp(-1, 1) + 1) / 2, nrow=nrow), path)

if __name__ == "__main__":
    torch.manual_seed(0)
    x = decode1(torch.randn(128, D, device=device))
    print("decode1:", x.shape, f"[{x.min():.2f},{x.max():.2f}] mean {x.mean():.3f}", flush=True)
    save_grid(x, f"{COMMON}/pymf_common_smoke.png")
    print("SMOKE DONE", flush=True)
