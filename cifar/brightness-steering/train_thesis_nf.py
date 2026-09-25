"""Normalising flow adapter for brightness steering, trained by reverse KL.

The observation is fixed at y* = 0.2 with sigma_y = 0.05, so the adapter takes no
conditioning input.

Architecture: the ml-tarflow Model with patch 4, giving 64 tokens of 48 dimensions, 4
blocks of 2 causal attention layers, 256 channels and affine couplings. The output
projection's weight and bias are zeroed, so q is exactly N(0, I) at step 0.

Objective: L = E_q[ log q(z) - log N(z) - r(f(z)) ]. The entropy term uses the
sticking-the-landing estimator, evaluating log q through a parameter-detached twin of the
flow. This removes the zero-mean score term, which carried 90.7 per cent of the gradient
variance on this task. Gradients reach the parameters only through the differentiable
sampler and the frozen flow map.

Adam 2e-4 decayed to 1e-5, batch 64, 6000 steps, gradient clipping at 5. The KL to the
prior is reported as a sample estimate, since the flow has no closed form for it.
"""
import copy, json, math, os, sys, time
import torch
import torch.nn.functional as F

# repo layout: shared modules live in common/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from flow_map import decode1, decode1_diff, save_grid, COMMON, D, device
sys.path.insert(0, f"{COMMON}/ml-tarflow")
from transformer_flow import Model

NAME = os.environ.get("NAME", "L1T_rkl_tarflow")
Y = float(os.environ.get("Y", "0.2"))
SIG = float(os.environ.get("SIG", "0.05"))
N_EVAL = int(os.environ.get("N_EVAL", "50000"))
LR = float(os.environ.get("LR", "2e-4"))
BATCH = int(os.environ.get("BATCH", "32"))
ACC = int(os.environ.get("ACC", "2"))  # gradient-accumulation micro-batches: eff batch = BATCH*ACC
STEPS = int(os.environ.get("STEPS", "6000"))
SEED = int(os.environ.get("SEED", "0"))
REF_PT = f"{COMMON}/L1T_ref.pt"
torch.manual_seed(SEED)
TOK, PDIM = 64, 48


def m_of(x):
    return x.mean(dim=(1, 2, 3))


def reward(x):
    return -(m_of(x) - Y) ** 2 / (2 * SIG ** 2)


def logN_img(z_img):
    return -0.5 * z_img.flatten(1).pow(2).sum(-1) - 0.5 * D * math.log(2 * math.pi)


# flow: unconditional TarFlow
q = Model(in_channels=3, img_size=32, patch_size=4, channels=256,
          num_blocks=4, layers_per_block=2, nvp=True, num_classes=0).to(device)
for b in q.blocks:
    torch.nn.init.zeros_(b.proj_out.bias)  # the reference zeroes only the weight, we zero the bias too for exact identity
n_par = sum(p.numel() for p in q.parameters())
print(f"{NAME}: unconditional TarFlow patch4 ch256 nb4x2 (head 64, mlp x4) params {n_par/1e6:.2f}M "
      f"| STL rkl, lr {LR} cosine, batch {BATCH}x{ACC}, {STEPS} steps | y*={Y} sigma_y={SIG}", flush=True)

_twin = copy.deepcopy(q)  # parameter-detached twin for the STL entropy term
for p in _twin.parameters():
    p.requires_grad_(False)


def logprob(model, z_img):
    """density direction (parallel): log q(z) = log N(u) + sum block log-dets (x D)."""
    x = model.patchify(z_img)
    logdet = torch.zeros(x.shape[0], device=device)
    for blk in model.blocks:
        x, ld = blk(x)  # reference MetaBlock.forward: ((x-b)e^{-a}, -a.mean([1,2]))
        logdet = logdet + ld * D
    return logN_img(model.unpatchify(x)) + logdet


def sample_diff(n):
    """differentiable sequential sampler (native orientation), no log-det bookkeeping
    (entropy handled by the STL detached density pass)."""
    x = torch.randn(n, TOK, PDIM, device=device)
    for blk in reversed(list(q.blocks)):
        xp = blk.permutation(x)
        pos = blk.permutation(blk.pos_embed, dim=0)
        T = xp.shape[1]
        for i in range(T - 1):
            mask = torch.tril(torch.ones(i + 1, i + 1, device=device))
            h = blk.proj_in(xp[:, :i + 1]) + pos[:i + 1]
            for ab in blk.attn_blocks:
                h = ab(h, mask)
            xa, xb = blk.proj_out(h)[:, i:i + 1].chunk(2, dim=-1)
            nxt = xp[:, i + 1:i + 2] * xa[:, 0:1].float().exp().type(xa.dtype) + xb[:, 0:1]
            xp = torch.cat([xp[:, :i + 1], nxt, xp[:, i + 2:]], dim=1)
        x = blk.permutation(xp, inverse=True)
    return q.unpatchify(x)


@torch.no_grad()
def sample(n, bs=1024):
    return torch.cat([sample_diff(min(bs, n - i)) for i in range(0, n, bs)])


# eval helpers
_incep = None

@torch.no_grad()
def incep_feats(x_cpu_f16, batch=256):
    global _incep
    if _incep is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        _incep = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        _incep.fc = torch.nn.Identity()
        _incep.eval().to(device)
        for p in _incep.parameters():
            p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    fs = []
    for i in range(0, x_cpu_f16.shape[0], batch):
        xb = x_cpu_f16[i:i+batch].to(device).float().clamp(-1, 1)
        xb = F.interpolate(xb, size=(299, 299), mode="bilinear", align_corners=False)
        fs.append(_incep(((xb + 1) / 2 - mean) / std).half().cpu())
    return torch.cat(fs)


@torch.no_grad()
def fid_from_feats(fa, fb):
    fa, fb = fa.float().to(device), fb.float().to(device)
    mu1, mu2 = fa.mean(0), fb.mean(0)
    c1 = (fa - mu1).T @ (fa - mu1) / (fa.shape[0] - 1)
    c2 = (fb - mu2).T @ (fb - mu2) / (fb.shape[0] - 1)
    s1, u1 = torch.linalg.eigh(c1.double())
    sq1 = (u1 * s1.clamp_min(0).sqrt()) @ u1.T
    ev = torch.linalg.eigvalsh(sq1 @ c2.double() @ sq1).clamp_min(0)
    return float(((mu1 - mu2) ** 2).sum().double() + c1.double().trace() + c2.double().trace() - 2 * ev.sqrt().sum())


ref = torch.load(REF_PT, map_location="cpu")
m_A_probe = ref["m_A"][torch.linspace(0, ref["m_A"].shape[0] - 1, 2048).long()].to(device)

# training
opt = torch.optim.Adam(q.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
best = {"w1": float("inf"), "step": 0}
t0 = time.time()
for i in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    _twin.load_state_dict(q.state_dict())  # refresh detached twin (no grad, ~ms)
    loss = 0.0
    for _a in range(ACC):  # eff batch BATCH*ACC (single graph per micro-batch)
        z = sample_diff(BATCH)
        lq = logprob(_twin, z)  # STL: params detached, gradient via z only
        l = ((lq - logN_img(z)).mean() - reward(decode1_diff(z.flatten(1))).mean()) / ACC
        l.backward()
        loss += float(l) * ACC
    torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
    opt.step(); sched.step()
    if not math.isfinite(loss):
        print(f"NaN at {i} - abort", flush=True); raise SystemExit(3)
    if i % 250 == 0:
        with torch.no_grad():
            zp = sample(2048)
            mp = m_of(decode1(zp.flatten(1)))
            w1p = float((mp.sort().values - m_A_probe.sort().values).abs().mean())
            zn = float(zp.flatten(1).norm(dim=1).mean())
            kl_est = float((logprob(q, zp[:512]) - logN_img(zp[:512])).mean())
        tag = ""
        if w1p < best["w1"] and 52 <= zn <= 58:
            best = {"w1": w1p, "step": i}
            torch.save(q.state_dict(), f"{COMMON}/{NAME}_best.pt")
            tag = " <-- best"
        print(f"{NAME} {i}: probe_w1 {w1p:.4f} m ({float(mp.mean()):.3f},{float(mp.std()):.3f}) "
              f"znorm {zn:.1f} KLest {kl_est:.1f} loss {loss:.1f} "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}G ({(time.time()-t0)/60:.1f}m){tag}", flush=True)
    if os.environ.get("SMOKE") == "1" and i >= 3:
        print(f"SMOKE OK (peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)", flush=True)
        raise SystemExit

torch.save(q.state_dict(), f"{COMMON}/{NAME}.pt")
# final 50k eval
with torch.no_grad():
    z = sample(N_EVAL).cpu()
    ms, xs = [], []
    for i in range(0, N_EVAL, 512):
        x = decode1(z[i:i+512].to(device).flatten(1))
        ms.append(m_of(x).cpu()); xs.append(x.half().cpu())
    m, ximg = torch.cat(ms), torch.cat(xs)
    feats = incep_feats(ximg)
    kl_est = float(sum((logprob(q, z[i:i+512].to(device)) - logN_img(z[i:i+512].to(device))).sum()
                       for i in range(0, 8192, 512)) / 8192)
    row = {"name": NAME, "y": Y, "sigma_y": SIG, "n_eval": N_EVAL, "params_M": round(n_par/1e6, 2),
           "w1_m": round(float((m.sort().values - ref["m_A"].sort().values).abs().mean()), 5),
           "fid_vs_A": round(fid_from_feats(feats, ref["feats_A"]), 3),
           "m_mean": round(float(m.mean()), 4), "m_std": round(float(m.std()), 4),
           "znorm": round(float(z.float().flatten(1).norm(dim=1).mean()), 2),
           "KL_est_nats": round(kl_est, 2), "floors": ref["floors"],
           "best_probe": {"w1": round(best["w1"], 5), "step": best["step"]}}
    save_grid(ximg[:128].float().to(device), f"{COMMON}/{NAME}_grid.png")
    json.dump(row, open(f"{COMMON}/{NAME}_row.json", "w"), indent=1)
    print(f"EVAL {NAME}: w1_m {row['w1_m']} (floor {row['floors']['w1_floor']}) "
          f"fid {row['fid_vs_A']} (floor {row['floors']['fid_floor']}) "
          f"m ({row['m_mean']},{row['m_std']}) znorm {row['znorm']} KLest {row['KL_est_nats']}", flush=True)
print(f"ARM {NAME} DONE ({(time.time()-t0)/60:.1f}m); best probe {best['w1']:.4f} @ {best['step']}", flush=True)
