"""Evaluate VFM inpainting checkpoints on 24 targets.

The probe run during training uses 8 targets. The frozen-map results are reported on 24,
made up of 16 generated images and 8 real ones, so this script reproduces that set and the
comparison is like for like.

Reports the residual through both the EMA and the raw flow map, hole diversity, the latent
norm, and the split between the generated and real halves.
"""
import os, sys, math, copy, json
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # repo layout: shared modules live in common/
from flow_map import COMMON  # importing it also puts py-meanflow on sys.path
device = "cuda"; D = 3 * 32 * 32; SIG = 0.05
BOX_LO, BOX_HI = 10, 22
_M = torch.ones(1, 1, 32, 32); _M[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI] = 0.0
MASK = _M.to(device)
N_OBS = int(MASK.sum().item()) * 3
RES_PERFECT = SIG * math.sqrt(N_OBS)
CKPT = sys.argv[1]; NAME = os.path.basename(CKPT).replace("VFM_", "").replace("_best.pt", "")

from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
_a = get_args_parser().parse_args([]); _a.use_edm_aug = True
sd = torch.load(CKPT, map_location="cpu")
net = instantiate_model(_a).net_ema1; net.load_state_dict(sd["net"]); net = net.to(device).eval()
net_ema = instantiate_model(_a).net_ema1; net_ema.load_state_dict(sd["net_ema"]); net_ema = net_ema.to(device).eval()
for m in (net, net_ema):
    for p in m.parameters(): p.requires_grad_(False)
u_of = lambda mod, x, t, h: mod(x, (t.view(-1), h.view(-1)), aug_cond=None)

import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(32, ch); self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(32, ch); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x))); h = self.c2(F.silu(self.n2(h))); return x + h
class GaussAdapter(nn.Module):
    def __init__(self, ch=256, nblocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(nblocks)])
        self.norm_out = nn.GroupNorm(32, ch); self.conv_out = nn.Conv2d(ch, 6, 3, padding=1)
    def forward(self, y):
        h = self.conv_in(y)
        for b in self.blocks: h = b(h)
        mu, lv = self.conv_out(F.silu(self.norm_out(h))).chunk(2, dim=1)
        return mu, lv.clamp(-10.0, 2.0)
adapter = GaussAdapter().to(device); adapter.load_state_dict(sd["adapter"]); adapter.eval()

# the 24 targets, identical construction to the INPff harness
from flow_map import decode1 as decode1_frozen
from torchvision import datasets
import torchvision.transforms.functional as TF
g = torch.Generator().manual_seed(999)
XG = decode1_frozen(torch.randn(16, D, generator=g).to(device))
YG = (XG + SIG * torch.randn(16, 3, 32, 32, generator=g).to(device)) * MASK
ds = datasets.CIFAR10(f"{COMMON}/cifar_data", train=False, download=False)
RI = [55, 3, 190, 421, 777, 1234, 2222, 4067]
xs = torch.stack([TF.to_tensor(ds[j][0]) * 2 - 1 for j in RI]).to(device)
gr = torch.Generator().manual_seed(997)
YR = (xs + SIG * torch.randn(8, 3, 32, 32, generator=gr).to(device)) * MASK
Y = torch.cat([YG, YR]); N_GEN = 16
import lpips
LP = lpips.LPIPS(net="alex").to(device)
hole = lambda x: F.interpolate(x[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI], size=48, mode="bilinear", align_corners=False)

rows = []
with torch.no_grad():
    for k in range(Y.shape[0]):
        yk = Y[k:k+1].expand(128, -1, -1, -1)
        mu, lv = adapter(yk); z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        ones = torch.ones(128, device=device)
        out = {}
        for tag, mod in (("ema", net_ema), ("raw", net)):
            xh = z - u_of(mod, z, ones, ones)
            out[tag] = (((xh - yk) * MASK) ** 2).flatten(1).sum(1).sqrt()
            if tag == "ema": xe = xh
        rows.append(dict(idx=k, real=k >= N_GEN,
                         res_ema=float(out["ema"].mean()), res_raw=float(out["raw"].mean()),
                         worst=float(out["ema"].max()),
                         div=float(LP(hole(xe[:64]).clamp(-1,1), hole(xe[64:]).clamp(-1,1)).mean()),
                         znorm=float(z.flatten(1).norm(dim=1).mean())))
agg = lambda key, sel=None: sum(r[key] for r in rows if sel is None or r["real"] == sel) / max(1, len([r for r in rows if sel is None or r["real"] == sel]))
print(f"\nEVAL24 {NAME}  (floor {RES_PERFECT:.3f}; frozen INPAINTING refs: best trained adapter 8.21, adapter+50 Landweber 3.24, RePaint 1.58)")
print(f"  res_ema {agg('res_ema'):.3f}  (gen {agg('res_ema', False):.3f}  real {agg('res_ema', True):.3f})")
print(f"  res_raw {agg('res_raw'):.3f}  (gen {agg('res_raw', False):.3f}  real {agg('res_raw', True):.3f})")
print(f"  worst {max(r['worst'] for r in rows):.3f}   div_hole {agg('div'):.4f}   znorm {agg('znorm'):.1f}", flush=True)
json.dump(dict(name=NAME, rows=rows,
               res_ema=agg('res_ema'), res_raw=agg('res_raw'), div=agg('div'),
               res_ema_gen=agg('res_ema', False), res_ema_real=agg('res_ema', True)),
          open(f"{COMMON}/VFM_{NAME}_eval24.json", "w"), indent=1)
print(f"saved VFM_{NAME}_eval24.json", flush=True)
