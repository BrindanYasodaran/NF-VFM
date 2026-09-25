"""Conditional TarFlow adapter for 4x super-resolution, in the IAF orientation.

Conditioning reuses the reference implementation's own mechanism. Its MetaBlocks add a
class embedding to the tokens after the input projection, and the observation embedding
takes that slot instead. Patch 4 over a 32x32 image gives 64 tokens, so the 8x8
observation maps one-to-one onto the token grid. The observation passes through a small
CNN, is flattened to 64 tokens, projected to the channel width by a per-block linear
layer, and added to every block's token stream. The CNN is shared across blocks.

The output projection's weight and bias are zeroed, so the flow is the identity at
initialisation for every observation.
"""
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # repo layout: shared modules live in common/
from flow_map import decode1_diff, decode1, save_grid, COMMON, D, device
sys.path.insert(0, f"{COMMON}/ml-tarflow")
from transformer_flow import Model

NAME = os.environ.get("NAME", "l4_tarflow")
STEPS = int(os.environ.get("STEPS", "1500"))
BATCH = int(os.environ.get("BATCH", "128"))
LR = float(os.environ.get("LR", "2e-4"))
CLIP = float(os.environ.get("CLIP", "5.0"))
SEED = int(os.environ.get("SEED", "0"))
CH = int(os.environ.get("CH", "256"))
NB = int(os.environ.get("NB", "4"))
LPB = int(os.environ.get("LPB", "2"))
DATA = os.environ.get("DATA", "real")
OBJ = os.environ.get("OBJ", "rkl")  # rkl (IAF) | npe (native MLE) | rkl_native (seq-sampled polish)
CKPT_INIT = os.environ.get("CKPT_INIT", "")
SIG = 0.05
SIGMULT = float(os.environ.get("SIGMULT", "0"))
FRAC = float(os.environ.get("FRAC", "0.45"))

def op(x):
    return F.interpolate(x, size=8, mode="bicubic", antialias=True)

def reward(x, y_img, sig=SIG):
    return -((op(x) - y_img) ** 2).flatten(1).sum(1) / (2 * sig ** 2)

def logN_img(z):
    return -0.5 * z.flatten(1).pow(2).sum(-1) - 0.5 * D * math.log(2 * math.pi)

torch.manual_seed(SEED)
q = Model(in_channels=3, img_size=32, patch_size=4, channels=CH,
          num_blocks=NB, layers_per_block=LPB, nvp=True, num_classes=0).to(device)
for b in q.blocks:
    nn.init.zeros_(b.proj_out.bias)

class CondPath(nn.Module):
    """Shared y-encoder + per-block token projections."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1))
        self.proj = nn.ModuleList([nn.Linear(64, CH) for _ in range(NB)])
    def forward(self, y_img):
        f = self.enc(y_img)  # (B,64,8,8)
        t = f.flatten(2).transpose(1, 2)  # (B,64tok,64)
        return [p(t) for p in self.proj]  # per-block (B,64,CH)

cond = CondPath().to(device)
if os.environ.get("CKPT_INIT", ""):
    _st = torch.load(os.environ["CKPT_INIT"], map_location=device)
    q.load_state_dict(_st["q"]); cond.load_state_dict(_st["cond"])
    print(f"warm-started from {os.environ['CKPT_INIT']}", flush=True)
n_par = sum(p.numel() for p in q.parameters()) + sum(p.numel() for p in cond.parameters())
print(f"{NAME}: cond-tarflow patch4 ch{CH} nb{NB}x{LPB} DATA={DATA} params {n_par/1e6:.2f}M lr={LR}", flush=True)

def block_forward_cond(block, x, ce):
    """Reference MetaBlock.forward with class_embed slot replaced by ce (B,T,CH)."""
    xp = block.permutation(x)
    pos = block.permutation(block.pos_embed, dim=0)
    x_in = xp
    h = block.proj_in(xp) + pos + block.permutation(ce, dim=1)
    for attn in block.attn_blocks:
        h = attn(h, block.attn_mask)
    h = block.proj_out(h)
    h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
    xa, xb = h.chunk(2, dim=-1)
    scale = (-xa.float()).exp().type(xa.dtype)
    out = block.permutation((x_in - xb) * scale, inverse=True)
    return out, -xa.mean(dim=[1, 2])

def q_logprob_native(z_img, y_img):
    """NPE training density: z tokens -> blocks FORWARD (native TarFlow MLE
    direction, parallel) -> u; logq(z|y) = logN(u) + sum logdets."""
    ces = cond(y_img)
    x = q.patchify(z_img)
    logdet = torch.zeros(x.shape[0], device=device)
    for blk, ce in zip(q.blocks, ces):
        x, ld = block_forward_cond(blk, x, ce)
        logdet = logdet + ld * D
    u = x
    logN_u = -0.5 * u.flatten(1).pow(2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
    return logN_u + logdet

def q_sample_native_diff(n, y_img):
    """Differentiable sequential conditional reverse; returns z and exact logq
    (logdet collected during generation: z_i = u_i*exp(xa_i)+xb_i)."""
    ces = cond(y_img)
    u = torch.randn(n, 64, 48, device=device)
    x = u
    logdet = torch.zeros(n, device=device)
    for blk, ce in zip(reversed(list(q.blocks)), reversed(ces)):
        xp = blk.permutation(x)
        pos = blk.permutation(blk.pos_embed, dim=0)
        cep = blk.permutation(ce, dim=1)
        T = xp.shape[1]
        for i in range(T - 1):
            h = blk.proj_in(xp[:, :i + 1]) + pos[:i + 1] + cep[:, :i + 1]
            mask = torch.tril(torch.ones(i + 1, i + 1, device=device))
            for attn in blk.attn_blocks:
                h = attn(h, mask)
            h = blk.proj_out(h)[:, i:i + 1]
            xa, xb = h.chunk(2, dim=-1)
            scale = xa[:, 0:1].float().exp().type(xa.dtype)
            xp = torch.cat([xp[:, :i + 1], xp[:, i + 1:i + 2] * scale + xb[:, 0:1],
                            xp[:, i + 2:]], dim=1)
            logdet = logdet + xa[:, 0].sum(-1)
        x = blk.permutation(xp, inverse=True)
    logN_u = -0.5 * u.flatten(1).pow(2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
    return q.unpatchify(x), logN_u - logdet

@torch.no_grad()
def q_sample_native(n, y_img):
    """Sampling for the native orientation: sequential conditional reverse
    (mirrors reference MetaBlock.reverse without KV cache: recompute prefix)."""
    ces = cond(y_img)
    x = torch.randn(n, 64, 48, device=device)
    for blk, ce in zip(reversed(list(q.blocks)), reversed(ces)):
        xp = blk.permutation(x)
        pos = blk.permutation(blk.pos_embed, dim=0)
        cep = blk.permutation(ce, dim=1)
        T = xp.shape[1]
        for i in range(T - 1):
            h = blk.proj_in(xp[:, :i + 1]) + pos[:i + 1] + cep[:, :i + 1]
            mask = torch.tril(torch.ones(i + 1, i + 1, device=device))
            for attn in blk.attn_blocks:
                h = attn(h, mask)
            h = blk.proj_out(h)[:, i:i + 1]
            xa, xb = h.chunk(2, dim=-1)
            xp = torch.cat([xp[:, :i + 1],
                            (xp[:, i + 1:i + 2] * xa[:, 0:1].float().exp().type(xa.dtype) + xb[:, 0:1]),
                            xp[:, i + 2:]], dim=1)
        x = blk.permutation(xp, inverse=True)
    return q.unpatchify(x)

def q_sample(n, y_img):
    ces = cond(y_img)
    u = torch.randn(n, 3, 32, 32, device=device)
    x = q.patchify(u)
    logdet = torch.zeros(n, device=device)
    for blk, ce in zip(q.blocks, ces):
        x, ld = block_forward_cond(blk, x, ce)
        logdet = logdet + ld * D
    return q.unpatchify(x), logN_img(u) - logdet

# task stream and eval
_train_real = None
def _real_pool():
    global _train_real
    if _train_real is None:
        from torchvision import datasets
        import torchvision.transforms.functional as TF
        ds = datasets.CIFAR10(f"{COMMON}/cifar_data", train=True, download=False)
        _train_real = torch.stack([TF.to_tensor(ds[j][0]) * 2 - 1
                                   for j in range(len(ds))]).to(device)
    return _train_real

def make_y(n, gen=None, source=None):
    source = source or DATA
    if source == "real" and gen is None:
        pool = _real_pool()
        xt = pool[torch.randint(0, pool.shape[0], (n,), device=device)]
    else:
        zt = torch.randn(n, D, device=device) if gen is None else torch.randn(n, D, generator=gen).to(device)
        xt = decode1(zt)
    y = op(xt) + SIG * torch.randn(n, 3, 8, 8, device=device)
    return y, xt

g_eval = torch.Generator().manual_seed(999)
Y_EVAL, X_EVAL = make_y(16, g_eval, source="gen")
sr4_tgt = torch.load(f"{COMMON}/pymf_SR4_target.pt", map_location=device)
Y_EVAL = torch.cat([Y_EVAL, sr4_tgt["y_star"].view(1, 3, 8, 8)])
X_EVAL = torch.cat([X_EVAL, sr4_tgt["x_true"]])
N_GEN = 16
REAL_IDX = [3, 190, 421, 777, 1234, 2222, 4067]
from torchvision import datasets
import torchvision.transforms.functional as TF
_ds = datasets.CIFAR10(f"{COMMON}/cifar_data", train=False, download=False)
xs_real = torch.stack([TF.to_tensor(_ds[j][0]) * 2 - 1 for j in REAL_IDX]).to(device)
g_rn = torch.Generator().manual_seed(997)
Y_EVAL = torch.cat([Y_EVAL, op(xs_real) + SIG * torch.randn(len(REAL_IDX), 3, 8, 8, generator=g_rn).to(device)])
X_EVAL = torch.cat([X_EVAL, xs_real])
N_EV = Y_EVAL.shape[0]

_lpips = None
def lp():
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device)
    return _lpips

def lpips_pairs(a, b):
    vals = []
    with torch.no_grad():
        for i in range(0, a.shape[0], 64):
            vals.append(lp()(a[i:i+64].clamp(-1, 1), b[i:i+64].clamp(-1, 1)).flatten())
    return torch.cat(vals)

@torch.no_grad()
def eval_y(idx, n=256):
    y = Y_EVAL[idx:idx+1].expand(n, -1, -1, -1)
    if OBJ in ("npe", "rkl_native"):
        z = q_sample_native(n, y)
    else:
        z, _ = q_sample(n, y)
    x = decode1(z.flatten(1))
    res = float(((op(x) - Y_EVAL[idx:idx+1]) ** 2).flatten(1).sum(1).sqrt().mean())
    lpt = float(lpips_pairs(x[:128], X_EVAL[idx:idx+1].expand(128, -1, -1, -1)).mean())
    div = float(lpips_pairs(x[:64], x[64:128]).mean())
    zn = float(z.flatten(1).norm(dim=1).mean())
    return res, lpt, div, zn, x

def save_labeled_rows(row_imgs, row_labels, path, per_row=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(row_imgs)
    fig, axes = plt.subplots(n, 1, figsize=(per_row * 0.9, n * 1.0))
    if n == 1:
        axes = [axes]
    for ax, imgs, lab in zip(axes, row_imgs, row_labels):
        strip = torch.cat([im for im in imgs[:per_row]], dim=2)
        ax.imshow(((strip.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy())
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

def row_label(i):
    if i == N_GEN:
        return "REAL test[55]\n(ship #55)"
    if i > N_GEN:
        return f"REAL test[{REAL_IDX[i - N_GEN - 1]}]"
    return f"GEN #{i}"

def final_eval(name):
    rows = []
    GRID_IDX = [0, 1, 2, 3, 16, 17, 18, 19]
    grids = []
    for i in range(N_EV):
        res, lpt, div, zn, x = eval_y(i)
        rows.append({"y_idx": i, "ship55": i == N_GEN, "real": i >= N_GEN,
                     "residual": round(res, 3), "lpips_true": round(lpt, 4),
                     "div": round(div, 4), "znorm": round(zn, 1)})
        if i in GRID_IDX:
            grids.append(x[:16])
    real_rows = [r for r in rows if r["y_idx"] >= N_GEN]
    gen_rows = [r for r in rows if r["y_idx"] < N_GEN]
    agg = {"name": name, "params_M": round(n_par / 1e6, 2), "flow": "cond-tarflow",
           "residual_mean_real": round(float(np.mean([r["residual"] for r in real_rows])), 3),
           "lpips_mean_real": round(float(np.mean([r["lpips_true"] for r in real_rows])), 4),
           "residual_mean_gen": round(float(np.mean([r["residual"] for r in gen_rows])), 3),
           "residual_mean": round(float(np.mean([r["residual"] for r in rows])), 3),
           "residual_worst": round(float(np.max([r["residual"] for r in rows])), 3),
           "residual_perfect": round(SIG * 192 ** 0.5, 3),
           "lpips_mean": round(float(np.mean([r["lpips_true"] for r in rows])), 4),
           "div_mean": round(float(np.mean([r["div"] for r in rows])), 4),
           "ship55": rows[N_GEN], "per_y": rows}
    save_grid(torch.cat(grids), f"{COMMON}/pymf_L4_{name}_grid.png")
    save_labeled_rows(grids, [row_label(i) for i in GRID_IDX],
                      f"{COMMON}/pymf_L4_{name}_grid_labeled.png")
    json.dump(agg, open(f"{COMMON}/pymf_L4_{name}_row.json", "w"), indent=1)
    print(f"EVAL {name}: mean_res {agg['residual_mean']} worst {agg['residual_worst']} "
          f"gen {agg['residual_mean_gen']} real {agg['residual_mean_real']} "
          f"lpips {agg['lpips_mean']} ship55 {rows[N_GEN]}", flush=True)
    return agg

with torch.no_grad():
    yt, _ = make_y(64)
    zi, lqi = q_sample(64, yt)
    print(f"init check: max|logq-logN| = {float((lqi - logN_img(zi)).abs().max()):.2e} "
          f"znorm {float(zi.flatten(1).norm(dim=1).mean()):.1f}", flush=True)

if os.environ.get("SMOKE", "0") == "1":
    STEPS = 10

params = list(q.parameters()) + list(cond.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
ANNEAL_END = int(FRAC * STEPS)
def sig_eff(i):
    if SIGMULT <= 0 or i >= ANNEAL_END:
        return SIG
    return SIG * SIGMULT ** (1.0 - i / ANNEAL_END)
t0 = time.time()
best = {"res": float("inf"), "step": 0}
for i in range(1, STEPS + 1):
    se = sig_eff(i)
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        y, _ = make_y(BATCH)
    if OBJ == "npe":
        with torch.no_grad():
            zt = torch.randn(BATCH, D, device=device)
            xt = decode1(zt)
            y = op(xt) + SIG * torch.randn(BATCH, 3, 8, 8, device=device)
        loss = (-q_logprob_native(zt.view(BATCH, 3, 32, 32), y)).mean()
    elif OBJ == "rkl_native":
        z, lq = q_sample_native_diff(BATCH, y)
        x = decode1_diff(z.flatten(1))
        loss = (lq - logN_img(z) - reward(x, y, se)).mean()
    else:
        z, lq = q_sample(BATCH, y)
        x = decode1_diff(z.flatten(1))
        loss = (lq - logN_img(z) - reward(x, y, se)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    opt.step(); sched.step()
    if not torch.isfinite(loss):
        print(f"NaN at {i} - abort", flush=True)
        raise SystemExit(3)
    if i % 250 == 0 or (os.environ.get("SMOKE") == "1" and i == STEPS):
        rs = [eval_y(j, 128)[0] for j in (0, 5, 11, N_GEN)]
        r = float(np.mean(rs))
        tag = ""
        if r < best["res"]:
            best = {"res": r, "step": i}
            torch.save({"q": q.state_dict(), "cond": cond.state_dict()},
                       f"{COMMON}/pymf_L4_{NAME}_best.pt")
            tag = " <-- best"
        print(f"{NAME} {i}: probe_res={r:.3f} (perfect ~0.693) per-y {[round(v,2) for v in rs]} "
              f"loss={loss.item():.1f} ({(time.time()-t0)/60:.1f}m){tag}", flush=True)

if os.environ.get("SMOKE", "0") == "1":
    print("SMOKE OK", flush=True)
    raise SystemExit
torch.save({"q": q.state_dict(), "cond": cond.state_dict()}, f"{COMMON}/pymf_L4_{NAME}.pt")
final_eval(NAME)
print(f"ARM {NAME} DONE ({(time.time()-t0)/60:.1f}m)", flush=True)
