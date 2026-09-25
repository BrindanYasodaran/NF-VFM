"""Calibrate the brightness task and build its reference sets.

Measures the distribution of mean pixel value under the flow map, picks y* and sigma_y so
that rejection sampling accepts between 5 and 12 per cent of proposals, draws 8192 exact
posterior latents by rejection with one-step decoding, and reports the metric floors.
"""
import json, os, sys
import torch

# repo layout: shared modules live in common/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from flow_map import decode1, save_grid, COMMON, D, device

torch.manual_seed(0)

def m_of(x):
    return x.mean(dim=(1, 2, 3))

zs = torch.randn(4096, D, device=device)
ms = m_of(decode1(zs))
mu, sd = float(ms.mean()), float(ms.std())
print(f"m under prior: mean {mu:.4f} std {sd:.4f} range [{float(ms.min()):.3f},{float(ms.max()):.3f}]", flush=True)

y_star = mu + 1.0 * sd
SIG = None
for sig in (0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03):
    acc = float(torch.exp(-(ms - y_star) ** 2 / (2 * sig ** 2)).mean())
    print(f"  sigma_y {sig:.2f}: predicted acceptance {acc:.4f}", flush=True)
    if SIG is None and 0.05 <= acc <= 0.12:
        SIG = sig
SIG = SIG or 0.06
print(f"CHOSEN: y* = {y_star:.4f}, sigma_y = {SIG}", flush=True)

def reward(x):
    return -(m_of(x) - y_star) ** 2 / (2 * SIG ** 2)

@torch.no_grad()
def reject(n, batch=4096):
    out, got, tried = [], 0, 0
    while got < n:
        z = torch.randn(batch, D, device=device)
        r = reward(decode1(z))
        a = torch.rand(batch, device=device) < torch.exp(r)
        out.append(z[a]); got += int(a.sum()); tried += batch
        assert tried < 400 * batch
    return torch.cat(out)[:n], got / tried

z_orc, acc = reject(8192)
x_orc = decode1(z_orc)
m_orc = m_of(x_orc)
print(f"oracle: acceptance {acc:.4f}; m mean {float(m_orc.mean()):.4f} std {float(m_orc.std()):.4f}", flush=True)
torch.save({"z": z_orc.cpu(), "y_star": y_star, "sigma_y": SIG}, f"{COMMON}/pymf_L1_oracle.pt")
save_grid(x_orc[:128], f"{COMMON}/pymf_L1_oracle_grid.png")
save_grid(decode1(zs[:128]), f"{COMMON}/pymf_L1_prior_grid.png")

a, b = m_orc[:4096].sort().values, m_orc[4096:].sort().values
w1_floor = float((a - b).abs().mean())
print(f"W1(m) split-half floor: {w1_floor:.5f}", flush=True)
json.dump({"y_star": y_star, "sigma_y": SIG, "acceptance": acc,
           "gen_m_mean": mu, "gen_m_std": sd, "w1_floor": w1_floor},
          open(f"{COMMON}/pymf_L1_setup.json", "w"), indent=1)
print("S1 DONE", flush=True)
