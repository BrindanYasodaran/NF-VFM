"""Gaussian adapters for brightness steering. The STAGE environment variable picks one of
three stages.

The measurement is the mean pixel value of the image, with y* = 0.2 and sigma_y = 0.05, so
the noise posterior is proportional to N(z; 0, I) exp(-(m(f(z)) - y*)^2 / 2 sigma_y^2).

Stage "oracle" draws 100k exact posterior latents by rejection and splits them into two
halves of 50k. One half is the reference that every method is scored against, and the other
is used to fit the Gaussians. Scoring one half against the other also gives the best score
any method could hope to reach.

Stage "mle" fits Gaussians in closed form to the fit half and scores them against the
reference half. It fits three: a diagonal covariance, a full sample covariance, and a full
covariance with Ledoit-Wolf shrinkage.

Stage "rkl" trains the adapter by reverse KL. A Gaussian has a closed form for the KL to
N(0, I), so only the reward term has to be sampled. Setting ARM=diag fits a mean and a log
standard deviation, and ARM=full fits a mean and a lower-triangular factor L, giving
covariance LL^T. Adam at 2e-4 decayed to 1e-5, batch 128, 6000 steps.
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F

# repo layout: shared modules live in common/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from flow_map import decode1, decode1_diff, save_grid, COMMON, D, device

STAGE = os.environ.get("STAGE", "oracle")
ARM = os.environ.get("ARM", "diag")  # rkl: diag | full
NAME = os.environ.get("NAME", f"L1T_{STAGE}_{ARM}")
Y = float(os.environ.get("Y", "0.2"))
SIG = float(os.environ.get("SIG", "0.05"))
N_ORC = int(os.environ.get("N_ORC", "100000"))
N_EVAL = int(os.environ.get("N_EVAL", "50000"))
LR = float(os.environ.get("LR", "2e-4"))
BATCH = int(os.environ.get("BATCH", "128"))
STEPS = int(os.environ.get("STEPS", "6000"))
SEED = int(os.environ.get("SEED", "0"))
ORC_PT = f"{COMMON}/L1T_oracle.pt"
REF_PT = f"{COMMON}/L1T_ref.pt"
torch.manual_seed(SEED)


def m_of(x):
    return x.mean(dim=(1, 2, 3))


def reward(x):
    return -(m_of(x) - Y) ** 2 / (2 * SIG ** 2)


# Inception features and FID
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
        xb = ((xb + 1) / 2 - mean) / std
        fs.append(_incep(xb).half().cpu())
    return torch.cat(fs)


@torch.no_grad()
def fid_from_feats(fa, fb):
    fa, fb = fa.float().to(device), fb.float().to(device)
    mu1, mu2 = fa.mean(0), fb.mean(0)
    c1 = (fa - mu1).T @ (fa - mu1) / (fa.shape[0] - 1)
    c2 = (fb - mu2).T @ (fb - mu2) / (fb.shape[0] - 1)
    diff = ((mu1 - mu2) ** 2).sum()
    s1, u1 = torch.linalg.eigh(c1.double())
    sq1 = (u1 * s1.clamp_min(0).sqrt()) @ u1.T
    ev = torch.linalg.eigvalsh(sq1 @ c2.double() @ sq1).clamp_min(0)
    return float(diff.double() + c1.double().trace() + c2.double().trace() - 2 * ev.sqrt().sum())


@torch.no_grad()
def w1_exact(a, b):
    """exact 1-D W1 = mean |sorted-a - sorted-b|; equal sizes required."""
    assert a.shape[0] == b.shape[0]
    return float((a.sort().values - b.sort().values).abs().mean())


@torch.no_grad()
def decode_stream(z, keep_images=True, bs=512):
    """decode latents in chunks; return (m values, fp16 CPU images or None)."""
    ms, xs = [], []
    for i in range(0, z.shape[0], bs):
        x = decode1(z[i:i+bs].float().to(device))
        ms.append(m_of(x).cpu())
        if keep_images:
            xs.append(x.half().cpu())
    return torch.cat(ms), (torch.cat(xs) if keep_images else None)


def load_ref():
    ref = torch.load(REF_PT, map_location="cpu")
    return ref  # m_A (50k), feats_A (50k,2048 fp16), floors dict


@torch.no_grad()
def full_eval(sample_fn, tag, kl_exact):
    """sample N_EVAL latents via sample_fn(n)->(n,D) on device; report vs reference A."""
    ref = load_ref()
    zs = []
    for i in range(0, N_EVAL, 4096):
        zs.append(sample_fn(min(4096, N_EVAL - i)).cpu())
    z = torch.cat(zs)
    m, ximg = decode_stream(z)
    feats = incep_feats(ximg)
    row = {"name": tag, "y": Y, "sigma_y": SIG, "n_eval": N_EVAL,
           "w1_m": round(w1_exact(m, ref["m_A"]), 5),
           "fid_vs_A": round(fid_from_feats(feats, ref["feats_A"]), 3),
           "m_mean": round(float(m.mean()), 4), "m_std": round(float(m.std()), 4),
           "znorm": round(float(z.float().norm(dim=1).mean()), 2),
           "KL_q_prior_nats": round(float(kl_exact), 2),
           "floors": ref["floors"]}
    save_grid(ximg[:128].float().to(device), f"{COMMON}/L1T_{tag}_grid.png")
    json.dump(row, open(f"{COMMON}/L1T_{tag}_row.json", "w"), indent=1)
    print(f"EVAL {tag}: w1_m {row['w1_m']} (floor {ref['floors']['w1_floor']}) "
          f"fid {row['fid_vs_A']} (floor {ref['floors']['fid_floor']}) "
          f"m ({row['m_mean']},{row['m_std']}) znorm {row['znorm']} KL {row['KL_q_prior_nats']}", flush=True)
    return row


# stage: oracle
if STAGE == "oracle":
    t0 = time.time()
    got, tried, out = 0, 0, []
    g = torch.Generator(device="cpu").manual_seed(999)
    while got < N_ORC:
        z = torch.randn(8192, D, generator=g).to(device)
        with torch.no_grad():
            r = reward(decode1(z))
            a = torch.rand(8192, device=device) < torch.exp(r)
        out.append(z[a].half().cpu()); got += int(a.sum()); tried += 8192
        if tried % (8192 * 20) == 0:
            print(f"oracle: {got}/{N_ORC} accepted ({got/max(tried,1):.4f}) {(time.time()-t0)/60:.1f}m", flush=True)
    z_all = torch.cat(out)[:N_ORC]
    acc = got / tried
    zA, zB = z_all[:N_ORC // 2], z_all[N_ORC // 2:]
    torch.save({"zA": zA, "zB": zB, "acc": acc, "y": Y, "sig": SIG}, ORC_PT)
    print(f"oracle: acceptance {acc:.4f} ({tried} tried), saved {ORC_PT}", flush=True)
    m_A, xA = decode_stream(zA)
    m_B, xB = decode_stream(zB)
    print(f"oracle m: A ({m_A.mean():.4f},{m_A.std():.4f}) B ({m_B.mean():.4f},{m_B.std():.4f}) "
          f"[linear-Gaussian prediction (0.190, 0.049)]", flush=True)
    feats_A, feats_B = incep_feats(xA), incep_feats(xB)
    floors = {"w1_floor": round(w1_exact(m_B, m_A), 5),
              "fid_floor": round(fid_from_feats(feats_B, feats_A), 3),
              "acceptance": round(acc, 4)}
    torch.save({"m_A": m_A, "feats_A": feats_A, "m_B": m_B, "floors": floors}, REF_PT)
    save_grid(xA[:128].float().to(device), f"{COMMON}/L1T_oracle_grid.png")
    json.dump({"acceptance": acc, "m_A": [float(m_A.mean()), float(m_A.std())],
               "m_B": [float(m_B.mean()), float(m_B.std())], "floors": floors,
               "znorm_A": float(zA.float().norm(dim=1).mean())},
              open(f"{COMMON}/L1T_oracle_stats.json", "w"), indent=1)
    print(f"ORACLE DONE: floors {floors} ({(time.time()-t0)/60:.1f}m)", flush=True)
    raise SystemExit

# stage: mle
if STAGE == "mle":
    zB = torch.load(ORC_PT, map_location="cpu")["zB"].float().to(device)  # fit set (50k, D)
    n = zB.shape[0]
    mu = zB.mean(0)
    Xc = zB - mu
    # diag
    sd = Xc.std(0, unbiased=True)
    kl_diag = 0.5 * float((mu ** 2 + sd ** 2 - 1 - 2 * sd.log()).sum())
    g = torch.Generator(device=device).manual_seed(7)
    full_eval(lambda k: mu + sd * torch.randn(k, D, device=device, generator=g), "mle_diag", kl_diag)
    # full (sample covariance)
    S = (Xc.T @ Xc) / (n - 1)
    # Ledoit-Wolf intensity: rho = b^2 / d^2 (LW 2004, identity-scaled target)
    m_tr = float(S.trace()) / D
    d2 = float(((S - m_tr * torch.eye(D, device=device)) ** 2).sum())
    x4 = float((Xc ** 2).sum(1).pow(2).sum()) / n ** 2
    b2 = min(d2, max(0.0, x4 - float((S ** 2).sum()) / n))
    rho = b2 / d2
    print(f"mle: tr(S)/d {m_tr:.4f}  LW rho {rho:.4f}", flush=True)
    for tag, Sm in (("mle_full", S), ("mle_full_lw", (1 - rho) * S + rho * m_tr * torch.eye(D, device=device))):
        for jit in (1e-6, 1e-4, 1e-2):
            try:
                L = torch.linalg.cholesky(Sm.double() + jit * torch.eye(D, device=device, dtype=torch.float64)).float()
                break
            except Exception:
                print(f"  cholesky jitter {jit:g} insufficient, escalating", flush=True)
        kl = 0.5 * (float((mu ** 2).sum()) + float(Sm.trace()) - D) - float(L.diagonal().log().sum())
        gg = torch.Generator(device=device).manual_seed(7)
        full_eval(lambda k, L=L: mu + torch.randn(k, D, device=device, generator=gg) @ L.T, tag, kl)
    print("MLE DONE", flush=True)
    raise SystemExit

# stage: rkl
class GaussDiag(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(D))
        self.log_std = torch.nn.Parameter(torch.zeros(D))
    def sample(self, k):
        return self.mu + self.log_std.clamp(-6, 3).exp() * torch.randn(k, D, device=device)
    def kl(self):  # KL(q || N(0,I)), closed form
        ls = self.log_std.clamp(-6, 3)
        return 0.5 * (self.mu ** 2 + (2 * ls).exp() - 1).sum() - ls.sum()

class GaussFull(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(D))
        self.diag_raw = torch.nn.Parameter(torch.zeros(D))  # diag(L) = exp(diag_raw)
        self.off = torch.nn.Parameter(torch.zeros(D, D))  # strictly-lower part of L
        self.register_buffer("mask", torch.tril(torch.ones(D, D), -1))
    def L(self):
        return self.off * self.mask + torch.diag(self.diag_raw.clamp(-6, 3).exp())
    def sample(self, k):
        return self.mu + torch.randn(k, D, device=device) @ self.L().T
    def kl(self):  # 0.5(||mu||^2 + tr(LL^T) - D) - sum log diag(L)
        L = self.L()
        return 0.5 * ((self.mu ** 2).sum() + (L ** 2).sum() - D) - self.diag_raw.clamp(-6, 3).sum()

q = (GaussDiag() if ARM == "diag" else GaussFull()).to(device)
n_par = sum(p.numel() for p in q.parameters())
print(f"{NAME}: rkl {ARM} gauss, {n_par/1e6:.3f}M params, exact-KL loss, "
      f"lr {LR} cosine, batch {BATCH}, {STEPS} steps", flush=True)
opt = torch.optim.Adam(q.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
ref = load_ref()
m_A_probe = ref["m_A"][torch.linspace(0, ref["m_A"].shape[0] - 1, 2048).long()].to(device)

best = {"w1": float("inf"), "step": 0}
t0 = time.time()
for i in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    z = q.sample(BATCH)
    loss = q.kl() - reward(decode1_diff(z)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
    opt.step(); sched.step()
    if not math.isfinite(float(loss)):
        print(f"NaN at {i} - abort", flush=True); raise SystemExit(3)
    if i % 250 == 0:
        with torch.no_grad():
            zp = q.sample(2048)
            mp = m_of(decode1(zp))
            w1p = float((mp.sort().values - m_A_probe.sort().values).abs().mean())
            zn = float(zp.norm(dim=1).mean())
        tag = ""
        if w1p < best["w1"] and 52 <= zn <= 58:
            best = {"w1": w1p, "step": i}
            torch.save(q.state_dict(), f"{COMMON}/L1T_rkl_{ARM}_best.pt")
            tag = " <-- best"
        print(f"{NAME} {i}: probe_w1 {w1p:.4f} m ({float(mp.mean()):.3f},{float(mp.std()):.3f}) "
              f"znorm {zn:.1f} KL {float(q.kl()):.1f} loss {float(loss):.1f} "
              f"({(time.time()-t0)/60:.1f}m){tag}", flush=True)

torch.save(q.state_dict(), f"{COMMON}/L1T_rkl_{ARM}.pt")
with torch.no_grad():
    full_eval(lambda k: q.sample(k), f"rkl_{ARM}", float(q.kl()))
print(f"ARM rkl_{ARM} DONE ({(time.time()-t0)/60:.1f}m); best probe {best['w1']:.4f} @ {best['step']}", flush=True)
