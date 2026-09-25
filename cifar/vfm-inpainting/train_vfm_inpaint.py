"""VFM for CIFAR-10 box inpainting: a Gaussian adapter trained jointly with the flow map.

The flow map is the py-meanflow CIFAR U-Net, initialised from net_ema1 and left trainable,
so both networks are updated from step 0. The operator zeroes a fixed central 12x12 box.
Measurement noise of 0.05 is added everywhere, which makes the term inside the hole a
constant that carries no gradient.

The loss is a port of the authors' ImageNet AdapterLoss and has three terms. The data term
is the residual on the observed pixels, decoded through an EMA copy of the flow map so that
it trains only the adapter. The KL term keeps the adapter close to N(0, I). The flow term
keeps the flow map a valid flow map, and is the mean of the flow matching and MeanFlow
losses, taken on interpolants whose noise endpoint is the adapter's z with probability ALPHA
and fresh noise otherwise. The three terms are summed and then rescaled by 1/(loss + 0.01).

Both networks use AdamW at 1e-4 with no weight decay. The paper uses alpha = 0.5 for images.

Evaluation reports the residual on the observed pixels, which has a floor of 2.569, the
diversity of the filled hole measured by LPIPS, and how far the flow map has drifted from
its pretrained state, measured by decoding a fixed batch of latents at every probe.
"""
import os, sys, math, time, copy
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))  # repo layout: shared modules live in common/
from flow_map import COMMON  # importing it also puts py-meanflow on sys.path
device = "cuda"
D = 3 * 32 * 32
NAME = os.environ.get("NAME", "vfm01")
ALPHA = float(os.environ.get("ALPHA", "0.5"))
BATCH = int(os.environ.get("BATCH", "64"))
STEPS = int(os.environ.get("STEPS", "3000"))
LR = float(os.environ.get("LR", "1e-4"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9999"))
DATA_W = float(os.environ.get("DATA_W", "1.0"))
KL_W = float(os.environ.get("KL_W", "1.0"))
FLOW_W = float(os.environ.get("FLOW_W", "1.0"))
ADAPTIVE = int(os.environ.get("ADAPTIVE", "1"))
CKPT_INIT = os.environ.get("CKPT_INIT", "")  # warm-start: restores net, net_ema, adapter
WARMUP = int(os.environ.get("WARMUP", "0"))  # linear LR warmup steps
COS_FROM = float(os.environ.get("COS_FROM", "1.0"))  # fraction of total steps at which cosine decay starts (1.0 = never)
LR_MIN = float(os.environ.get("LR_MIN", "1e-5"))  # cosine floor
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "0"))  # periodic checkpoints, 0 = off, final always saved
HFLIP = int(os.environ.get("HFLIP", "0"))  # random horizontal flip of training images
DRIFT_MAX = float(os.environ.get("DRIFT_MAX", "20")) # watchdog: stop if prior drift exceeds this
RES_MAX = float(os.environ.get("RES_MAX", "12"))  # watchdog: stop if res_raw > RES_MAX for 3 probes after step 3000
SIG = 0.05
BOX_LO, BOX_HI = 10, 22
_M = torch.ones(1, 1, 32, 32); _M[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI] = 0.0
MASK = _M.to(device)
N_OBS = int(MASK.sum().item()) * 3
RES_PERFECT = SIG * math.sqrt(N_OBS)  # 2.569

# flow map: trainable, with EMA theta^-
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser
_args = get_args_parser().parse_args([]); _args.use_edm_aug = True
_sd = torch.load(f"{COMMON}/pymf_ema1.pt", map_location="cpu")
net = instantiate_model(_args).net_ema1
net.load_state_dict(_sd); net = net.to(device).train()
for p in net.parameters(): p.requires_grad_(True)
net_ema = copy.deepcopy(net).eval()
for p in net_ema.parameters(): p.requires_grad_(False)

def u_of(module, x, t, h):
    return module(x, (t.view(-1), h.view(-1)), aug_cond=None)

# Gaussian adapter, in the authors' ResBlock style
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(32, ch); self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(32, ch); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h

class GaussAdapter(nn.Module):
    """y (3,32,32) -> mu, logvar (3,32,32). Zero-init out => starts exactly N(0, I)."""
    def __init__(self, ch=256, nblocks=8):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(nblocks)])
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, 6, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight); nn.init.zeros_(self.conv_out.bias)
    def forward(self, y):
        h = self.conv_in(y)
        for b in self.blocks: h = b(h)
        out = self.conv_out(F.silu(self.norm_out(h)))
        mu, lv = out.chunk(2, dim=1)
        return mu, lv.clamp(-10.0, 2.0)

adapter = GaussAdapter().to(device)
n_th = sum(p.numel() for p in net.parameters()); n_ad = sum(p.numel() for p in adapter.parameters())
print(f"{NAME}: VFM joint | flow map {n_th/1e6:.1f}M (trainable) | adapter {n_ad/1e6:.2f}M | "
      f"ALPHA={ALPHA} BATCH={BATCH} LR={LR} EMA={EMA_DECAY} floor={RES_PERFECT:.3f}", flush=True)

opt = torch.optim.AdamW(list(net.parameters()) + list(adapter.parameters()), lr=LR, weight_decay=0.0)
if CKPT_INIT:
    _ck = torch.load(CKPT_INIT, map_location="cpu")
    net.load_state_dict(_ck["net"]); net_ema.load_state_dict(_ck["net_ema"]); adapter.load_state_dict(_ck["adapter"])
    if "opt" in _ck:
        opt.load_state_dict(_ck["opt"]); print("  (optimizer state restored too)", flush=True)
    else:
        print("  (NO optimizer state in ckpt: a fresh Adam kicks every weight by ~LR on step 1 -- use WARMUP>=200)", flush=True)
    print(f"warm-started net/net_ema/adapter from {CKPT_INIT} (step {_ck.get('step','?')})", flush=True)

def lr_at(step):
    """linear warmup -> flat LR -> cosine from COS_FROM*STEPS down to LR_MIN at STEPS."""
    if WARMUP > 0 and step <= WARMUP:
        return LR * step / WARMUP
    s0 = int(COS_FROM * STEPS)
    if step <= s0:
        return LR
    f = (step - s0) / max(1, STEPS - s0)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * f))

# data
from torchvision import datasets
import torchvision.transforms.functional as TF
_ds = datasets.CIFAR10(f"{COMMON}/cifar_data", train=True, download=False)
_X = torch.stack([TF.to_tensor(_ds[j][0]) * 2 - 1 for j in range(len(_ds))])
def batch_x(n):
    x = _X[torch.randint(0, _X.shape[0], (n,))].to(device)
    if HFLIP:
        f = torch.rand(n, 1, 1, 1, device=device) < 0.5
        x = torch.where(f, x.flip(-1), x)
    return x

# eval targets (INPff conventions, masked-noise y, comparable numbers)
from flow_map import decode1 as decode1_frozen  # frozen reference decoder for target construction
g_eval = torch.Generator().manual_seed(999)
XG = decode1_frozen(torch.randn(16, D, generator=g_eval).to(device))
YG = (XG + SIG * torch.randn(16, 3, 32, 32, generator=g_eval).to(device)) * MASK
_dt = datasets.CIFAR10(f"{COMMON}/cifar_data", train=False, download=False)
xs = torch.stack([TF.to_tensor(_dt[j][0]) * 2 - 1 for j in [55, 3, 190, 421]]).to(device)
g_rn = torch.Generator().manual_seed(997)
YR = (xs + SIG * torch.randn(4, 3, 32, 32, generator=g_rn).to(device)) * MASK
Y_EVAL = torch.cat([YG[:4], YR]); X_EVAL = torch.cat([XG[:4], xs])
_lp = None
def lpips_pairs(a, b):
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net="alex").to(device)
    return _lp(a.clamp(-1, 1), b.clamp(-1, 1)).flatten()
hole_crop = lambda x: F.interpolate(x[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI], size=48, mode="bilinear", align_corners=False)

torch.manual_seed(0)
Z_GUARD = torch.randn(64, 3, 32, 32, device=device)
with torch.no_grad():
    X_GUARD0 = Z_GUARD - u_of(net_ema, Z_GUARD, torch.ones(64, device=device), torch.ones(64, device=device))

@torch.no_grad()
def probe(step):
    adapter.eval(); net.eval()
    ones = torch.ones(64, device=device)
    res_e, res_r, divs, zn, lpt = [], [], [], [], []
    for k in range(Y_EVAL.shape[0]):
        yk = Y_EVAL[k:k+1].expand(64, -1, -1, -1)
        xk = X_EVAL[k:k+1].expand(64, -1, -1, -1)
        mu, lv = adapter(yk)
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        zn.append(z.flatten(1).norm(dim=1).mean().item())
        for mod, acc in ((net_ema, res_e), (net, res_r)):
            xh = z - u_of(mod, z, ones, ones)
            acc.append((((xh - yk) * MASK) ** 2).flatten(1).sum(1).sqrt().mean().item())
            if mod is net:  # hole metrics on the RAW map -- the one we would deploy (EMA lags the adapter)
                divs.append(lpips_pairs(hole_crop(xh[:32]), hole_crop(xh[32:])).mean().item())
                lpt.append(lpips_pairs(hole_crop(xh), hole_crop(xk)).mean().item())  # hole vs truth: the metric
                                                                                          # the residual is blind to
    xg = Z_GUARD - u_of(net, Z_GUARD, ones, ones)
    drift = (xg - X_GUARD0).flatten(1).norm(dim=1).mean().item()
    if step % 500 == 0:
        from flow_map import save_grid
        save_grid(xg, f"{COMMON}/VFM_{NAME}_prior_grid.png", nrow=8)
    adapter.train(); net.train()
    return (sum(res_e)/8, sum(res_r)/8, sum(divs)/8, sum(zn)/8, drift, sum(lpt)/8)

# training
t0 = time.time(); best = float("inf")
WB = None
if os.environ.get("WANDB", "1") == "1":
    try:
        import wandb
        WB = wandb.init(project=os.environ.get("WANDB_PROJECT", "inpff"), name=NAME,
                        config=dict(alpha=ALPHA, batch=BATCH, lr=LR, ema=EMA_DECAY, steps=STEPS))
    except Exception as e:
        print(f"wandb off ({e})", flush=True)

CHUNK = int(os.environ.get("CHUNK", "16"))  # micro-batch: 3 graphs through the 55.9M U-Net
# (EMA data path, FM forward, MF JVP -- JVP doubles activations) OOM above ~16 on 24GB, so
# accumulate gradients over BATCH/CHUNK chunks and step once. adaptive() is per-sample, so
# chunking does not change the loss.
def loss_on(x):
    B = x.shape[0]
    # measurement, following the authors: op(x) then noise everywhere, so the hole term is constant wrt the parameters
    y = x * MASK + SIG * torch.randn_like(x)
    mu, lv = adapter(y)
    z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
    # 1. data consistency through EMA theta^- (grads reach adapter only)
    ones = torch.ones(B, device=device)
    xhat = z - u_of(net_ema, z, ones, ones)
    L_data = 200.0 * ((xhat * MASK - y) ** 2).mean(dim=(1, 2, 3))
    # 2. KL(N(mu, sig^2) || N(0, I)), mean over dims
    L_kl = (0.5 * (mu ** 2 + lv.exp() - lv - 1)).mean(dim=(1, 2, 3))
    # 3. flow losses on the eps mixture, z not detached as in the authors' code, same mix for FM and MF
    m = (torch.rand(B, 1, 1, 1, device=device) < ALPHA).float()
    eps_mix = m * z + (1 - m) * torch.randn_like(x)
    t_fm = torch.sigmoid(torch.randn(B, 1, 1, 1, device=device) * 1.0 - 0.4)
    x_t = (1 - t_fm) * x + t_fm * eps_mix
    v_t = eps_mix - x
    v_fm = u_of(net, x_t, t_fm, torch.zeros_like(t_fm))
    L_fm = ((v_fm - v_t.detach()) ** 2).mean(dim=(1, 2, 3))
    ln1 = torch.sigmoid(torch.randn(B, 1, 1, 1, device=device) * 1.0 - 0.2)
    ln2 = torch.sigmoid(torch.randn(B, 1, 1, 1, device=device) * 1.0 + 0.2)
    t_mf, r_mf = torch.maximum(ln1, ln2), torch.minimum(ln1, ln2)
    x_tm = (1 - t_mf) * x + t_mf * eps_mix
    v_tgt = (eps_mix - x).detach()
    def fwrap(xt, t, r): return u_of(net, xt, t, t - r)
    u, dudt = torch.func.jvp(fwrap, (x_tm, t_mf, r_mf),
                             (v_tgt, torch.ones_like(t_mf), torch.zeros_like(r_mf)))
    u_tgt = (v_tgt + (r_mf - t_mf) * dudt).detach()
    L_mf = ((u - u_tgt) ** 2).mean(dim=(1, 2, 3))
    L_flow = 0.5 * (L_fm + L_mf)
    loss = FLOW_W * L_flow + DATA_W * L_data + KL_W * L_kl  # per-sample
    if ADAPTIVE:
        w = 1.0 / (loss + 0.01)  # the authors' adaptive(), p=1 c=0.01
        loss = w.detach() * loss
    loss = loss.mean()
    return loss, L_data, L_kl, L_fm, L_mf, mu, lv

bad_probes = 0
for i in range(1, STEPS + 1):
    for g in opt.param_groups: g["lr"] = lr_at(i)
    opt.zero_grad(set_to_none=True)
    nch = max(1, BATCH // CHUNK)
    for _c in range(nch):
        loss, L_data, L_kl, L_fm, L_mf, mu, lv = loss_on(batch_x(CHUNK))
        (loss / nch).backward()
    opt.step()
    with torch.no_grad():
        for pe, p in zip(net_ema.parameters(), net.parameters()):
            pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
    if not math.isfinite(loss.item()):
        print(f"NaN at {i}", flush=True); raise SystemExit(3)
    if i % 50 == 0 and WB is not None:
        WB.log({"train/L_data": L_data.mean().item(), "train/L_kl": L_kl.mean().item(),
                "train/L_fm": L_fm.mean().item(), "train/L_mf": L_mf.mean().item(),
                "train/mu_abs": mu.abs().mean().item(), "train/sigma": (0.5*lv).exp().mean().item()}, step=i)
    if i % 250 == 0 or i == STEPS:
        re_, rr_, dv_, zn_, drift, lpt_ = probe(i)
        tag = ""
        if rr_ < best:  # select on the raw map, since res_ema is blind to the EMA hole artefact
            best = rr_
            torch.save({"net": net.state_dict(), "net_ema": net_ema.state_dict(),
                        "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": i}, f"{COMMON}/VFM_{NAME}_best.pt")
            tag = " <-- best"
        if CKPT_EVERY and i % CKPT_EVERY == 0:
            torch.save({"net": net.state_dict(), "net_ema": net_ema.state_dict(),
                        "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": i}, f"{COMMON}/VFM_{NAME}_step{i}.pt")
        print(f"{NAME} {i}: res_ema={re_:.3f} res_raw={rr_:.3f} (floor {RES_PERFECT:.3f}) "
              f"div={dv_:.3f} lpips_hole={lpt_:.3f} znorm={zn_:.1f} prior_drift={drift:.2f} lr={lr_at(i):.1e} "
              f"[data {L_data.mean():.3f} kl {L_kl.mean():.4f} fm {L_fm.mean():.3f} mf {L_mf.mean():.3f}] "
              f"({(time.time()-t0)/60:.1f}m){tag}", flush=True)
        if WB is not None:
            WB.log({"eval/res_ema": re_, "eval/res_raw": rr_, "eval/div": dv_, "eval/lpips_hole": lpt_,
                    "eval/znorm": zn_, "eval/prior_drift": drift, "train/lr": lr_at(i)}, step=i)
        # watchdogs
        stop = None
        if drift > DRIFT_MAX: stop = f"prior drift {drift:.1f} > {DRIFT_MAX} (prior collapse)"
        bad_probes = bad_probes + 1 if (i > 3000 and rr_ > RES_MAX) else 0
        if bad_probes >= 3: stop = f"res_raw > {RES_MAX} for 3 consecutive probes (divergence)"
        if stop:
            torch.save({"net": net.state_dict(), "net_ema": net_ema.state_dict(),
                        "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": i}, f"{COMMON}/VFM_{NAME}_stopped.pt")
            print(f"WATCHDOG STOP at {i}: {stop}", flush=True); break
torch.save({"net": net.state_dict(), "net_ema": net_ema.state_dict(), "adapter": adapter.state_dict(), "opt": opt.state_dict(), "step": i},
           f"{COMMON}/VFM_{NAME}_final.pt")
print(f"ARM {NAME} DONE ({(time.time()-t0)/60:.1f}m)", flush=True)
