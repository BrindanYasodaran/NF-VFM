"""Amortised inpainting with a conditional TarFlow noise adapter over a frozen flow map.

The observation is a CIFAR-10 image with a central 12x12 box removed, plus Gaussian noise
on the pixels that remain. The adapter learns q(z|y) over the frozen flow map's noise
space, so that decoding a sample gives an image that matches the observed pixels and
fills the hole plausibly. Setting FULLOBS=1 removes the box, leaving every pixel observed
and only one image consistent with each observation. That is the inversion diagnostic.

Objectives, selected by OBJ:
  npe         forward KL on exact (z, y) pairs drawn from the flow map
  rkl_native  reverse KL, sampling sequentially so that log q is exact
  rkl         reverse KL through the parallel direction (IAF orientation)
  mix         forward KL plus a reverse-KL term whose weight follows WSCHED

BETA weights the observation term of the reverse-KL loss. Raising it is equivalent to
assuming a smaller measurement noise, so the adapter targets a sharper posterior and
trades sample diversity for agreement with the observation.

CondPath turns the observation into one feature vector per flow token. COND chooses the
encoder, and FILM, INJ and XATTN are richer ways of injecting the result.

Evaluation uses 16 observations built from flow-map samples and 8 from real test images.
It reports the residual on the observed pixels, LPIPS to the truth, LPIPS between pairs
of samples over the hole as a measure of diversity, and the latent norm.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo layout: shared modules live in common/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from flow_map import decode1_diff, decode1, save_grid, COMMON, D, device
sys.path.insert(0, f"{COMMON}/ml-tarflow")
from transformer_flow import Model, AttentionBlock

NAME = os.environ.get("NAME", "ff_tarflow")
STEPS = int(os.environ.get("STEPS", "1500"))
BATCH = int(os.environ.get("BATCH", "128"))
LR = float(os.environ.get("LR", "2e-4"))
CLIP = float(os.environ.get("CLIP", "5.0"))
SEED = int(os.environ.get("SEED", "0"))
CH = int(os.environ.get("CH", "256"))
NB = int(os.environ.get("NB", "4"))
LPB = int(os.environ.get("LPB", "2"))
PATCH = int(os.environ.get("PATCH", "4"))
# observations built from real train images, or from flow-map samples
DATA = os.environ.get("DATA", "real")
OBJ = os.environ.get("OBJ", "rkl")  # rkl | rkl_native | npe | mix
CKPT_INIT = os.environ.get("CKPT_INIT", "")
# observation noise, setting both the reward weight and the perfect residual
SIG = float(os.environ.get("SIG", "0.05"))
BETA = float(os.environ.get("BETA", "1"))  # weight on the observation term of the reverse-KL loss
COND = os.environ.get("COND", "enc")  # enc (strided convs) | resctx (ResNet stem + attention)
RKLB = int(os.environ.get("RKLB", "32"))  # observations per step for the reverse-KL term of OBJ=mix

# OBJ=mix trains on L_fkl + w * L_rkl, where w follows WSCHED. Written as "step:w,step:w,...",
# it holds w at 0 until the first listed step and is piecewise constant after that, so training
# starts as pure maximum likelihood and the reverse-KL term is ramped in.
def parse_wsched(s):
    return sorted([(int(a), float(b)) for a, b in (kv.split(":") for kv in s.split(",") if kv)])
WSCHED = parse_wsched(os.environ.get("WSCHED", "1000:1e-3,1250:1e-2,1500:1e-1,1750:1"))
def w_of(step):
    w = 0.0
    for s, v in WSCHED:
        if step >= s:
            w = v
    return w

# FULLOBS=1 observes every pixel. The reward, the metrics and the encoder input are all
# derived from MASK, so dropping the box here is the whole change.
FULLOBS = os.environ.get("FULLOBS", "0") == "1"
BOX_LO, BOX_HI = 10, 22
_M = torch.ones(1, 1, 32, 32)
if not FULLOBS:
    _M[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI] = 0.0
MASK = _M.to(device)  # (1,1,32,32): 1 = observed, 0 = hole
N_OBS = int(MASK.sum().item()) * 3  # 880 px * 3 ch = 2640 observed values
RES_PERFECT = SIG * math.sqrt(N_OBS)  # 2.569

def op(x):
    return x * MASK

def reward(x, y_img):
    return -((op(x) - y_img) ** 2).flatten(1).sum(1) / (2 * SIG ** 2)

def logN_img(z):
    """log N(z; 0, I) for z of any shape with a leading batch dimension."""
    return -0.5 * z.flatten(1).pow(2).sum(-1) - 0.5 * D * math.log(2 * math.pi)

torch.manual_seed(SEED)
TOK = (32 // PATCH) ** 2  # tokens
PDIM = 3 * PATCH * PATCH  # dims per token
q = Model(in_channels=3, img_size=32, patch_size=PATCH, channels=CH,
          num_blocks=NB, layers_per_block=LPB, nvp=True, num_classes=0).to(device)
for _bi, b in enumerate(q.blocks):
    nn.init.zeros_(b.proj_out.bias)
    b._bi = _bi

class ResBlock(nn.Module):
    """VFM adapter ResBlock (models/adapter.py) minus the FiLM class conditioning:
    GN-SiLU-conv3x3-GN-SiLU-conv3x3 + skip."""
    def __init__(self, ch):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h

CTX_CH = int(os.environ.get("CTX_CH", "128"))  # resctx: context width
CTX_L = int(os.environ.get("CTX_L", "2"))      # resctx: full-attention layers
CTX_D = int(os.environ.get("CTX_D", "1"))      # resctx: stem ResBlocks per stage

# By default each block adds the observation feature to its token embeddings. The next three
# settings are richer alternatives, and can be combined. All start at the identity.

# FiLM scales and shifts the conditioner's hidden state instead of adding to it.
# "in" modulates the block input, "layers" also modulates between attention layers.
FILM = os.environ.get("FILM", "")
# An affine injector acts on the flow variable itself rather than on the hidden state, as in
# Glow. "z" puts one before the first block, "all" one before every block.
INJ = os.environ.get("INJ", "")
# Cross-attention lets each flow token attend over all the encoder features, rather than
# receiving a single feature vector.
XATTN = os.environ.get("XATTN", "0") == "1"

class CrossAttn(nn.Module):
    """Pre-norm cross-attention from the flow tokens (B,T,CH) onto the encoder tokens
    (B,TOK,fdim). The output projection is zero-init, so this starts as the identity."""
    def __init__(self, ch, fdim, head_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.q = nn.Linear(ch, ch)
        self.kv = nn.Linear(fdim, 2 * ch)
        self.proj = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.nh = ch // head_dim
    def forward(self, h, e):
        B, T, C = h.shape
        q = self.q(self.norm(h.float()).type(h.dtype)).reshape(B, T, self.nh, -1).transpose(1, 2)
        k, v = self.kv(e).reshape(B, e.shape[1], 2 * self.nh, -1).transpose(1, 2).chunk(2, dim=1)
        # unmasked: y is fully observed, so attending over it freely does not break the
        # autoregressive order over z
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, T, C))

class CondPath(nn.Module):
    """Turns the observation into one feature vector per flow token, which each block receives
    in the slot TarFlow uses for a class embedding.

    COND=enc encodes y with strided convolutions down to an 8x8 grid of 64 features.
    COND=resctx uses a deeper ResNet stem followed by full attention over the 64 tokens, and
    concatenates each token's own raw pixels. Its wider receptive field gives the tokens
    covering the hole some global context."""
    def __init__(self):
        super().__init__()
        assert PATCH == 4, "both encoders build an 8x8 feature grid, so they are patch-4 only"
        self.enc = None
        if COND == "enc":
            self.enc = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.SiLU(),  # 32 -> 16
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),  # 16 -> 8
                nn.Conv2d(64, 64, 3, padding=1))  # 8 -> 8
            fdim = 64
        elif COND == "resctx":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                *[ResBlock(32) for _ in range(CTX_D)],
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
                *[ResBlock(64) for _ in range(CTX_D)],
                nn.Conv2d(64, CTX_CH, 3, stride=2, padding=1),  # 16 -> 8
                *[ResBlock(CTX_CH) for _ in range(2 * CTX_D)])
            self.ctx_pos = nn.Parameter(torch.randn(TOK, CTX_CH) * 1e-2)
            self.ctx_blocks = nn.ModuleList([AttentionBlock(CTX_CH, 32, 4) for _ in range(CTX_L)])
            fdim = PDIM + CTX_CH
        else:
            raise ValueError(f"unknown COND: {COND}")
        if FILM:
            self.film = nn.ModuleList([
                nn.ModuleList([nn.Linear(fdim, 2 * CH) for _ in range(1 if FILM == "in" else LPB + 1)])
                for _ in range(NB)])
            for blk_ms in self.film:
                for m in blk_ms:
                    m.weight.data[:CH].zero_(); m.bias.data[:CH].zero_()  # gamma half zero-init
        else:
            self.proj = nn.ModuleList([nn.Linear(fdim, CH) for _ in range(NB)])
        if INJ:
            self.inj = nn.ModuleList([nn.Linear(fdim, 2 * PDIM) for _ in range(1 if INJ == "z" else NB)])
            for m in self.inj:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)  # (s, b) start at the identity
        if XATTN:
            self.xa_pos = nn.Parameter(torch.randn(TOK, fdim) * 1e-2)
            self.xattn = nn.ModuleList([
                nn.ModuleList([CrossAttn(CH, fdim) for _ in range(LPB)]) for _ in range(NB)])
    def forward(self, y_img):
        if self.enc is not None:
            f = self.enc(y_img)  # (B,64,8,8)
            t = f.flatten(2).transpose(1, 2)  # (B,64tok,64): token r*8+c <-> feature (r,c)
        else:
            t = q.patchify(y_img)  # (B,64tok,48): same patch order as z
            f = self.stem(y_img)  # (B,CTX_CH,8,8)
            h = f.flatten(2).transpose(1, 2) + self.ctx_pos
            for blk in self.ctx_blocks:
                h = blk(h, None)  # attn_mask=None -> full (bidirectional) attention
            t = torch.cat([t, h], dim=-1)  # (B,64tok,48+CTX_CH)
        inj = None
        if INJ:
            inj = []
            for m in self.inj:
                s, b = m(t).chunk(2, dim=-1)
                inj.append((s.clamp(-10, 2), b))
        global XA_TOKENS
        XA_TOKENS = (t + self.xa_pos) if XATTN else None
        if FILM:
            return [[m(t) for m in blk_ms] for blk_ms in self.film], inj
        return [p(t) for p in self.proj], inj  # per-block (B,64,CH)

XA_TOKENS = None
cond = CondPath().to(device)

if CKPT_INIT:
    _st = torch.load(CKPT_INIT, map_location=device)
    q.load_state_dict(_st["q"])
    cond.load_state_dict(_st["cond"])
    print(f"warm-started from {CKPT_INIT}", flush=True)
n_par = sum(p.numel() for p in q.parameters()) + sum(p.numel() for p in cond.parameters())
print(f"{NAME}: cond-tarflow patch{PATCH} ch{CH} nb{NB}x{LPB} COND={COND}"
      f"{'+film-' + FILM if FILM else ''}{'+inj-' + INJ if INJ else ''}{'+xattn' if XATTN else ''}"
      f" DATA={DATA} OBJ={OBJ} beta={BETA:g} params {n_par/1e6:.2f}M lr={LR} "
      f"perfect_res={RES_PERFECT:.3f}", flush=True)

def _ckpt_dict():
    return dict(q=q.state_dict(), cond=cond.state_dict())

# wandb is optional: WANDB=0 disables it, and a missing login never blocks the run.
WB = None
def wb_log(data, step=None):
    if WB is not None:
        WB.log(data, step=step)

def wb_init():
    global WB
    if os.environ.get("WANDB", "1") == "0" or os.environ.get("SMOKE", "0") == "1":
        return
    try:
        import wandb
        if wandb.api.api_key is None and "WANDB_API_KEY" not in os.environ:
            print("wandb: no API key found (run `wandb login`) - continuing without", flush=True)
            return
        WB = wandb.init(project=os.environ.get("WANDB_PROJECT", "inpff"), name=NAME,
                        config={"steps": STEPS, "batch": BATCH, "rklb": RKLB, "lr": LR,
                                "clip": CLIP, "seed": SEED, "ch": CH, "nb": NB, "lpb": LPB,
                                "cond": COND, "obj": OBJ, "data": DATA, "beta": BETA,
                                "ctx_ch": CTX_CH, "ctx_l": CTX_L, "ctx_d": CTX_D,
                                "film": FILM, "inj": INJ, "xattn": XATTN, "patch": PATCH,
                                "fullobs": FULLOBS, "wsched": os.environ.get("WSCHED", ""),
                                "params_M": round(n_par / 1e6, 2)})
        WB.log({"ref/truth": wandb.Image(f"{COMMON}/INPff_ref_truth.png"),
                "ref/observation": wandb.Image(f"{COMMON}/INPff_ref_observation.png")}, step=0)
    except Exception as e:
        print(f"wandb disabled ({e}) - continuing without", flush=True)

def perm_ce(block, ce):
    """Apply the block's token permutation to the conditioning (a list of sites if FILM)."""
    if FILM:
        return [block.permutation(t, dim=1) for t in ce]
    return block.permutation(ce, dim=1)

def film_mod(h, gb):
    g, b = gb.chunk(2, dim=-1)
    return (1 + g) * h + b

def attn_layer(block, li, h, mask):
    """One AttentionBlock, with a cross-attention residual inserted between the self-attention
    and the MLP when XATTN is set."""
    ab = block.attn_blocks[li]
    if not XATTN:
        return ab(h, mask)
    h = h + ab.attention(h, mask)
    h = h + cond.xattn[block._bi][li](h, XA_TOKENS)
    h = h + ab.mlp(h)
    return h

def cond_attn(block, h, ce_p, mask):
    """Attention stack with conditioning, shared by the density direction and both samplers.

    The samplers pass a token prefix rather than the full sequence. Conditioning is pointwise
    in the token index, so slicing it to h's length keeps every direction consistent."""
    T = h.shape[1]
    if FILM:
        h = film_mod(h, ce_p[0][:, :T])  # site 0: block input
        for li in range(len(block.attn_blocks)):
            if FILM == "layers" and li > 0:
                h = film_mod(h, ce_p[li][:, :T])  # between attention layers
            h = attn_layer(block, li, h, mask)
        if FILM == "layers":
            h = film_mod(h, ce_p[len(block.attn_blocks)][:, :T])  # before proj_out
    else:
        h = h + ce_p[:, :T]
        for li in range(len(block.attn_blocks)):
            h = attn_layer(block, li, h, mask)
    return h

def block_forward_cond(block, x, ce):
    """Reference MetaBlock.forward with the class-embedding slot replaced by ce."""
    xp = block.permutation(x)
    pos = block.permutation(block.pos_embed, dim=0)
    h = cond_attn(block, block.proj_in(xp) + pos, perm_ce(block, ce), block.attn_mask)
    ho = block.proj_out(h)
    ho = torch.cat([torch.zeros_like(ho[:, :1]), ho[:, :-1]], dim=1)
    xa, xb = ho.chunk(2, dim=-1)
    scale = (-xa.float()).exp().type(xa.dtype)
    out = block.permutation((xp - xb) * scale, inverse=True)
    return out, -xa.mean(dim=[1, 2])

# The flow runs in two directions. Going from z to the base variable u is parallel over tokens
# and yields the density, which is what maximum likelihood needs. Coming back from u to z has to
# be sequential, one token at a time, because each token's affine parameters are computed from
# the tokens before it. Reverse KL has to sample, so it pays that cost.

def flow_forward_tokens(x, ces, inj):
    """z tokens -> u tokens, applying each injector and then its block. Returns (u, logdet)."""
    logdet = torch.zeros(x.shape[0], device=device)
    for k, (blk, ce) in enumerate(zip(q.blocks, ces)):
        if inj is not None and k < len(inj):
            s, b = inj[k]
            x = (x - b) * torch.exp(-s)
            logdet = logdet - s.flatten(1).sum(-1)
        x, ld = block_forward_cond(blk, x, ce)
        logdet = logdet + ld * D
    return x, logdet

def q_logprob_native(z_img, y_img):
    """log q(z|y) by the parallel direction: log N(u) plus the accumulated logdet."""
    ces, inj = cond(y_img)
    u, logdet = flow_forward_tokens(q.patchify(z_img), ces, inj)
    return logN_img(u) + logdet

@torch.no_grad()
def q_sample_native(n, y_img, u=None):
    """Draw z ~ q(z|y) by the sequential reverse. Mirrors the reference MetaBlock.reverse
    without a KV cache, so every step recomputes the prefix."""
    ces, inj = cond(y_img)
    x = torch.randn(n, TOK, PDIM, device=device) if u is None else u
    for k, (blk, ce) in enumerate(zip(reversed(list(q.blocks)), reversed(ces))):
        xp = blk.permutation(x)
        pos = blk.permutation(blk.pos_embed, dim=0)
        cep = perm_ce(blk, ce)
        T = xp.shape[1]
        for i in range(T - 1):
            mask = torch.tril(torch.ones(i + 1, i + 1, device=device))
            h = cond_attn(blk, blk.proj_in(xp[:, :i + 1]) + pos[:i + 1], cep, mask)
            ho = blk.proj_out(h)[:, i:i + 1]
            xa, xb = ho.chunk(2, dim=-1)
            nxt = xp[:, i + 1:i + 2] * xa[:, 0:1].float().exp().type(xa.dtype) + xb[:, 0:1]
            xp = torch.cat([xp[:, :i + 1], nxt, xp[:, i + 2:]], dim=1)
        x = blk.permutation(xp, inverse=True)
        site = NB - 1 - k  # site index of the gap before this block
        if inj is not None and site < len(inj):
            s, b = inj[site]
            x = x * torch.exp(s) + b
    return q.unpatchify(x)

def q_sample_native_diff(n, y_img):
    """As q_sample_native, but differentiable and also returning log q. The logdet is
    accumulated as the tokens are generated, so no second pass is needed."""
    ces, inj = cond(y_img)
    u = torch.randn(n, TOK, PDIM, device=device)
    x = u
    logdet = torch.zeros(n, device=device)
    for k, (blk, ce) in enumerate(zip(reversed(list(q.blocks)), reversed(ces))):
        xp = blk.permutation(x)
        pos = blk.permutation(blk.pos_embed, dim=0)
        cep = perm_ce(blk, ce)
        T = xp.shape[1]
        for i in range(T - 1):
            mask = torch.tril(torch.ones(i + 1, i + 1, device=device))
            h = cond_attn(blk, blk.proj_in(xp[:, :i + 1]) + pos[:i + 1], cep, mask)
            ho = blk.proj_out(h)[:, i:i + 1]
            xa, xb = ho.chunk(2, dim=-1)
            scale = xa[:, 0:1].float().exp().type(xa.dtype)
            nxt = xp[:, i + 1:i + 2] * scale + xb[:, 0:1]
            xp = torch.cat([xp[:, :i + 1], nxt, xp[:, i + 2:]], dim=1)
            logdet = logdet + xa[:, 0].sum(-1)
        x = blk.permutation(xp, inverse=True)
        site = NB - 1 - k  # injector of the gap before this block
        if inj is not None and site < len(inj):
            s, b = inj[site]
            x = x * torch.exp(s) + b
            logdet = logdet + s.flatten(1).sum(-1)
    return q.unpatchify(x), logN_img(u) - logdet

def q_sample(n, y_img):
    """Draw z by applying the density direction to a base sample. Parallel and so much cheaper,
    but this is the IAF orientation, not the direction the flow is trained in."""
    ces, inj = cond(y_img)
    u = torch.randn(n, TOK, PDIM, device=device)
    x, logdet = flow_forward_tokens(u, ces, inj)
    return q.unpatchify(x), logN_img(u) - logdet

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
    """n observations and the images that produced them. source=real draws CIFAR train
    images, source=gen decodes fresh prior noise through the flow map."""
    source = source or DATA
    if source == "real" and gen is None:
        pool = _real_pool()
        xt = pool[torch.randint(0, pool.shape[0], (n,), device=device)]
    else:
        zt = torch.randn(n, D, device=device) if gen is None else torch.randn(n, D, generator=gen).to(device)
        xt = decode1(zt)
    noise = torch.randn(n, 3, 32, 32, device=device) if gen is None else torch.randn(n, 3, 32, 32, generator=gen).to(device)
    y = op(xt) + SIG * MASK * noise
    return y, xt

# Eval targets: 16 observations from flow-map samples, then 8 from real test images.
g_eval = torch.Generator().manual_seed(999)
Y_EVAL, X_EVAL = make_y(16, g_eval, source="gen")
N_GEN = 16
REAL_IDX = [55, 3, 190, 421, 777, 1234, 2222, 4067]  # test[55] = ship #55 first
from torchvision import datasets
import torchvision.transforms.functional as TF
_ds = datasets.CIFAR10(f"{COMMON}/cifar_data", train=False, download=False)
xs_real = torch.stack([TF.to_tensor(_ds[j][0]) * 2 - 1 for j in REAL_IDX]).to(device)
g_rn = torch.Generator().manual_seed(997)
Y_EVAL = torch.cat([Y_EVAL, op(xs_real) + SIG * MASK * torch.randn(len(REAL_IDX), 3, 32, 32, generator=g_rn).to(device)])
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

def hole_crop(x):
    """12x12 hole crop, upsampled to 48x48 so LPIPS(alex) has enough resolution."""
    return F.interpolate(x[:, :, BOX_LO:BOX_HI, BOX_LO:BOX_HI], size=48, mode="bilinear", align_corners=False)

def eval_y(idx, n=256):
    y = Y_EVAL[idx:idx+1].expand(n, -1, -1, -1)
    with torch.no_grad():
        z = q_sample_native(n, y)  # eval always uses the sequential sampler
        return _eval_metrics(idx, z, y)

@torch.no_grad()
def _eval_metrics(idx, z, y):
    x = decode1(z.flatten(1))
    res = float(((op(x) - Y_EVAL[idx:idx+1]) ** 2).flatten(1).sum(1).sqrt().mean())
    xt = X_EVAL[idx:idx+1]
    lpt = float(lpips_pairs(x[:128], xt.expand(128, -1, -1, -1)).mean())
    lph = float(lpips_pairs(hole_crop(x[:128]), hole_crop(xt).expand(128, -1, -1, -1)).mean())
    div = float(lpips_pairs(hole_crop(x[:64]), hole_crop(x[64:128])).mean())
    zn = float(z.flatten(1).norm(dim=1).mean())
    zf = z.flatten(1)[:64]
    zspread = float((zf[:32] - zf[32:64]).norm(dim=1).mean())  # spread of the latents for one y
    return res, lpt, lph, div, zn, x, zspread

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
        return f"REAL test[{REAL_IDX[i - N_GEN]}]"
    return f"GEN #{i}"

GRID_IDX = [0, 1, 2, 3, 16, 17, 18, 19]

def save_reference_strips():
    """Once per experiment: ground truth and masked observation for the 8 grid targets."""
    for tag, imgs in (("truth", X_EVAL), ("observation", Y_EVAL)):
        p = f"{COMMON}/INPff_ref_{tag}.png"
        if not os.path.exists(p):
            save_labeled_rows([imgs[i:i+1] for i in GRID_IDX], [row_label(i) for i in GRID_IDX], p, per_row=1)

def outline_hole(x, color=(1.0, -1.0, -1.0)):
    """Outline the hole in red, one pixel outside it so that no hole content is covered."""
    x = x.clone()
    lo, hi = BOX_LO - 1, BOX_HI  # 9 and 22 for the 12x12 box at 10..21
    for c, v in enumerate(color):
        x[:, c, lo, lo:hi + 1] = v; x[:, c, hi, lo:hi + 1] = v
        x[:, c, lo:hi + 1, lo] = v; x[:, c, lo:hi + 1, hi] = v
    return x

# grey: hole masked as in y | outline: red 1-px outline
HOLEVIEW = os.environ.get("HOLEVIEW", "grey")
def hole_view(x):
    return outline_hole(x) if HOLEVIEW == "outline" else x * MASK

def save_boxed_rows(row_imgs, idxs, path, n_samples=14):
    """One row per target: ground truth, observation, a gap, then the samples. By default the
    samples have the hole masked out just as the observation does, which makes the fit on the
    observed pixels easy to compare tile by tile."""
    rows, labels = [], []
    gap = torch.ones(1, 3, 32, 2, device=device)  # 2-px white separator
    for imgs, i in zip(row_imgs, idxs):
        ref = torch.cat([X_EVAL[i:i+1], Y_EVAL[i:i+1], gap], dim=3)  # (1,3,32,66)
        smp = torch.cat([im for im in hole_view(imgs[:n_samples])], dim=2)  # (3,32,32*n)
        rows.append(torch.cat([ref[0], smp], dim=2).unsqueeze(0))  # concat along width
        labels.append(row_label(i) + "\ntruth | y | samples")
    save_labeled_rows(rows, labels, path, per_row=(66 + 32 * n_samples) // 32 + 1)

GRID_EVERY = int(os.environ.get("GRID_EVERY", "1000"))  # in-training sample-grid cadence (steps)
@torch.no_grad()
def probe_grid(step):
    """Sample grid during training: 8 fixed targets by 16 samples. All 128 go through the
    sequential sampler as one batch, so they cost no more steps than a single sample would.
    The file is overwritten each time, with the history kept on wandb."""
    y = torch.cat([Y_EVAL[i:i+1].expand(16, -1, -1, -1) for i in GRID_IDX])
    z = q_sample_native(y.shape[0], y)
    x = decode1(z.flatten(1))
    rows = [x[k*16:(k+1)*16] for k in range(len(GRID_IDX))]
    p = f"{COMMON}/INPff_{NAME}_probegrid.png"
    save_labeled_rows(rows, [row_label(i) for i in GRID_IDX], p)
    pb = f"{COMMON}/INPff_{NAME}_probegrid_boxed.png"
    save_boxed_rows(rows, GRID_IDX, pb, n_samples=min(14, rows[0].shape[0]))
    if WB is not None:
        import wandb
        WB.log({"samples/grid": wandb.Image(p, caption=f"step {step}"),
                "samples/grid_boxed": wandb.Image(pb, caption=f"step {step}")}, step=step)

def final_eval(name):
    rows = []
    grids = []
    for i in range(N_EV):
        res, lpt, lph, div, zn, x, zsp = eval_y(i)
        rows.append({"y_idx": i, "ship55": i == N_GEN, "real": i >= N_GEN,
                     "residual": round(res, 3), "lpips_true": round(lpt, 4),
                     "lpips_hole": round(lph, 4), "div_hole": round(div, 4), "znorm": round(zn, 1),
                     "zspread": round(zsp, 1)})
        if i in GRID_IDX:
            grids.append(x[:16])
    real_rows = [r for r in rows if r["y_idx"] >= N_GEN]
    gen_rows = [r for r in rows if r["y_idx"] < N_GEN]
    agg = {"name": name, "params_M": round(n_par / 1e6, 2), "flow": "cond-tarflow", "cond": COND,
           "obj": OBJ, "beta": BETA, "fullobs": FULLOBS,
           "residual_mean_real": round(float(np.mean([r["residual"] for r in real_rows])), 3),
           "lpips_mean_real": round(float(np.mean([r["lpips_true"] for r in real_rows])), 4),
           "residual_mean_gen": round(float(np.mean([r["residual"] for r in gen_rows])), 3),
           "residual_mean": round(float(np.mean([r["residual"] for r in rows])), 3),
           "residual_worst": round(float(np.max([r["residual"] for r in rows])), 3),
           "residual_perfect": round(RES_PERFECT, 3),
           "lpips_mean": round(float(np.mean([r["lpips_true"] for r in rows])), 4),
           "lpips_hole_mean": round(float(np.mean([r["lpips_hole"] for r in rows])), 4),
           "div_hole_mean": round(float(np.mean([r["div_hole"] for r in rows])), 4),
           "znorm_mean": round(float(np.mean([r["znorm"] for r in rows])), 1),
           "ship55": rows[N_GEN], "per_y": rows}
    save_grid(torch.cat(grids), f"{COMMON}/INPff_{name}_grid.png")
    save_boxed_rows(grids, GRID_IDX, f"{COMMON}/INPff_{name}_grid_boxed.png")
    save_labeled_rows(grids, [row_label(i) for i in GRID_IDX],
                      f"{COMMON}/INPff_{name}_grid_labeled.png")
    json.dump(agg, open(f"{COMMON}/INPff_{name}_row.json", "w"), indent=1)
    if WB is not None:
        import wandb
        WB.log({f"final/{k}": v for k, v in agg.items() if isinstance(v, (int, float))})
        WB.log({"final/grid": wandb.Image(f"{COMMON}/INPff_{name}_grid_labeled.png")})
    print(f"EVAL {name}: mean_res {agg['residual_mean']} worst {agg['residual_worst']} "
          f"gen {agg['residual_mean_gen']} real {agg['residual_mean_real']} "
          f"lpips {agg['lpips_mean']} lpips_hole {agg['lpips_hole_mean']} div_hole {agg['div_hole_mean']} "
          f"znorm {agg['znorm_mean']} ship55 {rows[N_GEN]}", flush=True)
    return agg

save_reference_strips()
if os.environ.get("EVAL_ONLY", "0") == "1":
    assert CKPT_INIT, "EVAL_ONLY needs CKPT_INIT"
    print(f"EVAL_ONLY {NAME}", flush=True)
    final_eval(NAME)
    raise SystemExit

# Startup checks: the flow is the identity at initialisation, the sequential sampler and the
# density direction invert each other and agree on log q, and the reward ignores the hole.
with torch.no_grad():
    yt, xt = make_y(64)
    zi, lqi = q_sample(64, yt)
    print(f"init check: max|logq-logN| = {float((lqi - logN_img(zi)).abs().max()):.2e} "
          f"znorm {float(zi.flatten(1).norm(dim=1).mean()):.1f}", flush=True)
    u0 = torch.randn(4, TOK, PDIM, device=device)
    z0 = q_sample_native(4, yt[:4], u=u0)
    ces0, inj0 = cond(yt[:4])
    x0, _ = flow_forward_tokens(q.patchify(z0), ces0, inj0)
    print(f"check forward(sample(u)) = u: max|du| = {float((x0 - u0).abs().max()):.2e}", flush=True)
    zs, lqs = q_sample_native_diff(2, yt[:2])
    lqd = q_logprob_native(zs, yt[:2])
    print(f"check logq(sampler) = logq(density): max|dlogq| = {float((lqs - lqd).abs().max()):.2e} (|logq| ~ {float(lqd.abs().mean()):.0f})", flush=True)
    r0 = reward(xt, yt)
    xh = xt + (1 - MASK) * torch.randn_like(xt)
    r1 = reward(xh, yt)
    print(f"check hole-blindness: max|r(x)-r(x+hole noise)| = {float((r0 - r1).abs().max()):.2e}; "
          f"mean residual of the true x on its own y ={float(((op(xt) - yt) ** 2).flatten(1).sum(1).sqrt().mean()):.3f} "
          f"(perfect {RES_PERFECT:.3f})", flush=True)

if os.environ.get("SMOKE", "0") == "1":
    STEPS = 10

params = list(q.parameters()) + list(cond.parameters())

def mix_terms():
    """The two terms of OBJ=mix, built from one batch of flow-map samples. The forward-KL term
    uses the exact (z, y) pairs, and the reverse-KL term the first RKLB of the same
    observations."""
    with torch.no_grad():
        zt = torch.randn(BATCH, D, device=device)
        xt = decode1(zt)
        y = op(xt) + SIG * MASK * torch.randn(BATCH, 3, 32, 32, device=device)
    L_fkl = (-q_logprob_native(zt.view(BATCH, 3, 32, 32), y)).mean()
    y_s = y[:RKLB]
    z, lq = q_sample_native_diff(RKLB, y_s)
    x = decode1_diff(z.flatten(1))
    L_rkl = (lq - logN_img(z) - BETA * reward(x, y_s)).mean()
    return L_fkl, L_rkl

def grad_norm_of(loss, retain=True):
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))

if os.environ.get("DIAG", "0") == "1":
    # Compare the two gradient norms at the loaded checkpoint, and report the share the
    # reverse-KL term would take at each candidate weight. Used to pick WSCHED.
    gf, gr = [], []
    for _ in range(int(os.environ.get("DIAG_N", "4"))):
        L_fkl, L_rkl = mix_terms()
        gf.append(grad_norm_of(L_fkl))
        gr.append(grad_norm_of(L_rkl, retain=False))
        print(f"  L_fkl {L_fkl.item():.1f} |g_fkl| {gf[-1]:.2f}   L_rkl {L_rkl.item():.1f} |g_rkl| {gr[-1]:.2f}", flush=True)
    gf, gr = float(np.mean(gf)), float(np.mean(gr))
    print(f"DIAG mean |g_fkl| {gf:.2f}  |g_rkl| {gr:.2f}  ratio {gr/gf:.3g}", flush=True)
    for w in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        print(f"  w={w:g}: rkl share = {w*gr/(gf + w*gr):.3f}", flush=True)
    raise SystemExit

opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
wb_init()
t0 = time.time()
best = {"res": float("inf"), "step": 0}
PROBE_IDX = (0, 5, 11, N_GEN)
PROBE_EVERY = 250
# Keep the best checkpoint by residual, but only among those that still have healthy diversity
# and a latent norm near the prior's. Late in training the residual keeps improving as diversity
# collapses, so without the guard the best-scoring checkpoint is the most collapsed one.
DIV_FLOOR = float(os.environ.get("DIV_FLOOR", "0.20"))
ZN_LO = float(os.environ.get("ZN_LO", "52"))
ZN_HI = float(os.environ.get("ZN_HI", "58"))
for i in range(1, STEPS + 1):
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        y, _xg = make_y(BATCH)
    if OBJ == "npe":
        with torch.no_grad():
            zt = torch.randn(BATCH, D, device=device)
            xt = decode1(zt)
            y = op(xt) + SIG * MASK * torch.randn(BATCH, 3, 32, 32, device=device)
        loss = (-q_logprob_native(zt.view(BATCH, 3, 32, 32), y)).mean()
    elif OBJ == "rkl_native":
        z, lq = q_sample_native_diff(BATCH, y)
        x = decode1_diff(z.flatten(1))
        loss = (lq - logN_img(z) - BETA * reward(x, y)).mean()
    elif OBJ == "mix":
        w = w_of(i)
        probe_step = (i % PROBE_EVERY == 0) or (os.environ.get("SMOKE") == "1" and i == STEPS)
        if w > 0 or probe_step:
            L_fkl, L_rkl = mix_terms()
            if probe_step:  # per-term gradient norms: extra backward passes, probe steps only
                gnf = grad_norm_of(L_fkl)
                gnr = grad_norm_of(L_rkl, retain=w > 0)
                gshare = w * gnr / (gnf + w * gnr) if w > 0 else 0.0
                mix_log = (L_fkl.item(), L_rkl.item(), gnf, gnr, gshare)
            loss = L_fkl + (w * L_rkl if w > 0 else 0.0)
            del L_fkl, L_rkl  # keep only the floats: do not hold the sampler graph across steps
        else:
            with torch.no_grad():
                zt = torch.randn(BATCH, D, device=device)
                xt = decode1(zt)
                y = op(xt) + SIG * MASK * torch.randn(BATCH, 3, 32, 32, device=device)
            loss = (-q_logprob_native(zt.view(BATCH, 3, 32, 32), y)).mean()
    else:
        z, lq = q_sample(BATCH, y)
        x = decode1_diff(z.flatten(1))
        loss = (lq - logN_img(z) - BETA * reward(x, y)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, CLIP)
    opt.step(); sched.step()
    loss_val = loss.item(); del loss
    if not math.isfinite(loss_val):
        print(f"NaN at {i} - abort", flush=True)
        raise SystemExit(3)
    if OBJ == "mix" and w_of(i) == 0 and w_of(i + 1) > 0:
        # end of the pure forward-KL phase, before the reverse-KL term switches on
        torch.save(_ckpt_dict(), f"{COMMON}/INPff_{NAME}_npe_end.pt")
        print(f"{NAME} {i}: saved NPE-end checkpoint (w becomes {w_of(i+1):g} next step)", flush=True)
    if i % PROBE_EVERY == 0 or (os.environ.get("SMOKE") == "1" and i == STEPS):
        ev = [eval_y(j, 128) for j in PROBE_IDX]
        zsp_m = float(np.mean([e[6] for e in ev]))
        rs = [e[0] for e in ev]
        bs = [float(((op(e[5].mean(0, keepdim=True)) - Y_EVAL[j:j+1]) ** 2).sum().sqrt())
              for e, j in zip(ev, PROBE_IDX)]  # bias = ||A xbar - y|| per probe target
        r = float(np.mean(rs)); dv = float(np.mean([e[3] for e in ev])); zn = float(np.mean([e[4] for e in ev]))
        tag = ""
        guard_ok = dv >= DIV_FLOOR and ZN_LO <= zn <= ZN_HI
        if r < best["res"] and guard_ok:
            best = {"res": r, "step": i}
            torch.save(_ckpt_dict(), f"{COMMON}/INPff_{NAME}_best.pt")
            tag = " <-- best"
        elif r < best["res"]:
            tag = f" (res-best REJECTED by guard: div {dv:.3f} zn {zn:.1f})"
        extra = ""
        logd = {"probe/residual": r, "probe/bias": float(np.mean(bs)), "probe/div_hole": dv,
                "probe/znorm": zn, "train/loss": loss_val, "train/lr": sched.get_last_lr()[0]}
        for j, v, b in zip(PROBE_IDX, rs, bs):
            logd[f"probe/res_y{j}"] = v
            logd[f"probe/bias_y{j}"] = b
        if OBJ == "mix":
            lf, lr_, gnf, gnr, gshare = mix_log
            R = gnr / gnf if gnf > 0 else float("nan")
            extra = (f" | w={w_of(i):g} L_fkl={lf:.1f} L_rkl={lr_:.1f} "
                     f"|g_fkl|={gnf:.2f} |g_rkl|={gnr:.2f} R={R:.0f} rkl_share={gshare:.3f}")
            logd.update({"mix/w": w_of(i), "mix/L_fkl": lf, "mix/L_rkl": lr_,
                         "mix/g_fkl": gnf, "mix/g_rkl": gnr, "mix/R": R, "mix/share": gshare})
            for wv in sorted({v for _, v in WSCHED if v > 0}):
                logd[f"mix/share_hyp/w{wv:g}"] = wv * gnr / (gnf + wv * gnr)
        wb_log(logd, step=i)
        if i % GRID_EVERY == 0:
            probe_grid(i)
        print(f"{NAME} {i}: probe_res={r:.3f} bias={float(np.mean(bs)):.3f} (perfect {RES_PERFECT:.3f}) per-y {[round(v,2) for v in rs]} "
              f"div_hole={dv:.3f} zspread={zsp_m:.1f} znorm={zn:.1f} loss={loss_val:.1f} ({(time.time()-t0)/60:.1f}m){extra}{tag}", flush=True)

if os.environ.get("SMOKE", "0") == "1":
    print(f"SMOKE OK (peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)", flush=True)
    raise SystemExit
torch.save(_ckpt_dict(), f"{COMMON}/INPff_{NAME}.pt")
final_eval(NAME)
print(f"ARM {NAME} DONE ({(time.time()-t0)/60:.1f}m)", flush=True)
