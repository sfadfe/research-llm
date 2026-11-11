# Gate-recording interface (for Eval/): set `mixer.record_gate = True` on a GatedDeltaNet (default False, zero cost
# when off). Each forward then stores `mixer.last_gate = {"a", "b", "s", "c", "gamma"}` (detached): a, b = raw gate
# logits exactly as passed to chunk_gated_delta_rule (shape (B,T,H); (1,B*T,H) with cu), s = bounded scores (B,T,H)
# or None (linear / pool_s0), c = pooled c_t before RMSNorm, float32 (B,T,H,pool_rank) or None (linear),
# gamma = per-head gamma float32 (H,) (zeros for pool_g0) or None (linear). Recording uses the eager pooling path.
# Kernel gates: alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b). Mode: config [model].gate_mode.
# gate_mode "span" (okkie-attention, spans + lifetime gate): see okkie_span.py; `mixer.span` holds its parameters.
# Recording in span mode stores a, b and w (lifetime gate, (B,T,H) float32); s, c, gamma are None.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedLinearCrossEntropyLoss, FusedRMSNormGated, RMSNorm
from fla.modules.causal_conv1d import causal_conv1d
from fla.ops.attn import parallel_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.nsa import parallel_nsa

from okkie_span import OkkieSpan
from pool_triton import pool_gate_triton


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 24
    pattern: str = "GGGS"
    mlp_hidden: int = 2048
    norm_eps: float = 1e-6
    gdn_heads: int = 6
    gdn_head_dim: int = 128
    conv_size: int = 4
    nsa_q_heads: int = 16
    nsa_kv_heads: int = 1
    nsa_head_dim: int = 64
    nsa_block_size: int = 64
    nsa_block_counts: int = 8
    nsa_window: int = 256
    init_std: float = 0.02
    loss: str = "fused"
    # gate_mode: "linear" (A: [a,b] = ab(x)), "pool" (B: score + gamma learned),
    # "pool_g0" (C: gamma fixed 0), "pool_s0" (D: score fixed 0)
    gate_mode: str = "linear"
    pool_chunk: int = 64
    pool_ckpt: bool = True
    pool_compile: bool = True
    pool_impl: str = "triton"  # "triton" (fused kernels, CUDA) | "torch" (pool_gate; compiled if pool_compile)
    pool_h_min: float = 4.0
    pool_h_max: float = 256.0
    pool_rank: int = 16  # r: pooled x_i = A_h [k_i; v_i], A_h (r, 2*head_dim)
    score_bound: float = 10.0
    # gate_mode "span" (okkie_span.py); also uses pool_chunk (lifetime-gate m_h update period) and pool_ckpt
    span_hidden: int = 256  # feature MLP hidden width
    span_dim: int = 32  # per-head feature dim P (z_t has gdn_heads * span_dim dims)
    span_h_short: float = 4.0  # horizon of the short causal exponential average of z
    span_h_long: float = 64.0  # horizon of the long one
    span_threshold: float = 0.5  # theta: a span starts when cos(short, long) falls below theta
    span_max: int = 32  # span ids clamped to span_max - 1
    mung_dim: int = 128  # per-head pooled [k; v] dim
    span_top_n: int = 2  # usable spans read through the restore layer
    span_m: int = 4  # (k, v) pairs per restored span

    def kinds(self):
        return [self.pattern[i % len(self.pattern)] for i in range(self.n_layers)]


GATE_MODES = ("linear", "pool", "pool_g0", "pool_s0", "span")


def segment_ids(B, T, cu, device):
    """Document id per position, nondecreasing along T. cu (flattened, B == 1) or None (each row one doc)."""
    if cu is None:
        return torch.zeros(B, T, dtype=torch.long, device=device)
    assert B == 1
    flag = torch.zeros(T, dtype=torch.long, device=device)
    flag[cu[1:-1].long()] = 1
    return flag.cumsum(0)[None]


def padded_segments(B, T, cu, chunk, device):
    """segment_ids padded to a multiple of chunk; pad positions get their own id (after all real ones)."""
    seg = segment_ids(B, T, cu, device)
    pad = -(-T // chunk) * chunk - T
    if pad:
        seg = torch.cat([seg, (seg[:, -1:] + 1).expand(B, pad)], 1)
    return seg


def pool_chunks(z, s, g, seg, C):
    """Static-shape core of decay_pool. z (B,T,H,D), s (B,T,H), g (H,) float32; seg (B,Tp) with Tp % C == 0.
    Intra-chunk (C x C) decay matrix + chunk-level (nc x nc) decay matrix for the carry across chunks.
    Every exponent is <= max(s) (masked entries are set to -inf before exp): no overflow for any gamma >= 0."""
    B, T, H, D = z.shape
    Tp = seg.shape[1]
    nc = Tp // C
    dev = z.device
    if Tp > T:
        z = F.pad(z, (0, 0, 0, 0, 0, Tp - T))
        s = F.pad(s, (0, 0, 0, Tp - T))
    zc = z.reshape(B, nc, C, H, D).permute(0, 3, 1, 2, 4)
    sc = s.reshape(B, nc, C, H).permute(0, 3, 1, 2)
    segc = seg.reshape(B, 1, nc, C)
    l = torch.arange(C, device=dev)
    dist = (l[:, None] - l[None, :]).float()
    mask = (dist >= 0) & (segc[..., :, None] == segc[..., None, :])
    gam = g.view(1, H, 1, 1, 1)
    W = torch.exp((sc[..., None, :] - gam * dist).masked_fill(~mask, float("-inf")))
    num = W @ zc
    den = W.sum(-1)
    if nc > 1:
        n = torch.arange(nc, device=dev)
        cd = (n[:, None] - 1 - n[None, :]).float()
        first, last = segc[..., 0], segc[..., -1]
        cmask = (cd >= 0) & (first[..., :, None] == last[..., None, :])
        Pm = torch.exp((-(gam[..., 0] * C) * cd).masked_fill(~cmask, float("-inf")))
        Sn = Pm @ num[..., -1, :]
        Sd = Pm @ den[..., -1:]
        f = torch.exp(-gam[..., 0] * (l + 1).float()) * (segc == first[..., None])
        num = num + f[..., None] * Sn[..., None, :]
        den = den + f * Sd
    out = num / den[..., None]
    return out.permute(0, 2, 3, 1, 4).reshape(B, Tp, H, D)[:, :T]


def decay_pool(z, s, gamma, cu=None, chunk=64):
    """Causal recency-weighted softmax pooling, reset at document starts.

    c_t = sum_{i<=t, doc(i)==doc(t)} exp(s_i - gamma*(t-i)) z_i / sum_{...} exp(s_i - gamma*(t-i))
    z: (B,T,H,D); s: (B,T,H) or None (=0); gamma: (H,) >= 0 or None (=0). Returns float32 (B,T,H,D).
    """
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        B, T, H, D = z.shape
        dev = z.device
        s = torch.zeros(B, T, H, device=dev) if s is None else s.float()
        g = torch.zeros(H, device=dev) if gamma is None else gamma.float()
        return pool_chunks(z.float(), s, g, padded_segments(B, T, cu, chunk, dev), chunk)


def pool_gate(k, v, seg, proj_w, score_w, gamma_raw, norm_w, chunk, bound, eps, ret_c=False):
    """k, v (B,T,H,dk), proj_w (H*r, 2dk) -> RMSNorm_head(c) flattened (B,T,H*r) float32, the scores s (or None)
    [, c, gamma]. Pools x = A_h [k; v] (r dims); the score is computed from [k; v]."""
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        z = torch.cat([k, v], -1).float()
        B, T, H, D = z.shape
        R = norm_w.shape[0]
        x = torch.einsum("bthd,hrd->bthr", z, proj_w.float().view(H, R, D))
        if score_w is None:
            s = None
            sz = torch.zeros(B, T, H, device=z.device)
        else:
            s = sz = bound * torch.tanh(torch.einsum("bthd,hd->bth", z, score_w.float()) / bound)
        g = torch.zeros(H, device=z.device) if gamma_raw is None else F.softplus(gamma_raw.float())
        c = pool_chunks(x, sz, g, seg, chunk)
        cn = F.rms_norm(c, (R,), norm_w.float(), eps).flatten(-2)
        return (cn, s, c, g) if ret_c else (cn, s)


pool_gate_compiled = torch.compile(pool_gate, dynamic=False)


class GatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.dk = c.gdn_heads, c.gdn_head_dim
        kd = self.h * self.dk
        self.qkv = nn.Linear(c.d_model, 3 * kd, bias=False)
        self.ab = nn.Linear(c.d_model, 2 * self.h, bias=False)
        self.g_proj = nn.Linear(c.d_model, kd, bias=False)
        self.o_proj = nn.Linear(kd, c.d_model, bias=False)
        bound = c.conv_size ** -0.5
        self.conv = nn.Parameter(torch.empty(3 * kd, c.conv_size).uniform_(-bound, bound))
        self.A_log = nn.Parameter(torch.empty(self.h).uniform_(0, 16).log())
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm = FusedRMSNormGated(self.dk, eps=c.norm_eps, activation="swish")
        # pooled-gate extras: created without touching the global RNG so shared params match arm A
        assert c.gate_mode in GATE_MODES, c.gate_mode
        self.mode = c.gate_mode
        self.pool = self.mode not in ("linear", "span")
        self.record_gate = False
        self.last_gate = None
        if self.mode == "span":
            self.span = OkkieSpan(c, self.h, self.dk)
        if self.pool:
            self.chunk, self.ckpt, self.bound, self.compile_pool = c.pool_chunk, c.pool_ckpt, c.score_bound, c.pool_compile
            self.impl = c.pool_impl
            self.h_min, self.h_max = c.pool_h_min, c.pool_h_max
            dz, self.rank = 2 * self.dk, c.pool_rank
            self.c_proj = nn.Parameter(torch.empty(self.h * self.rank, dz))
            self.ab_c = nn.Parameter(torch.zeros(2 * self.h, self.h * self.rank))
            self.c_norm = nn.RMSNorm(self.rank, eps=c.norm_eps)
            self.score_w = nn.Parameter(torch.zeros(self.h, dz)) if self.mode != "pool_s0" else None
            self.gamma_raw = nn.Parameter(torch.zeros(self.h)) if self.mode != "pool_g0" else None
            self.compiled = False
            self.reset_pool(torch.Generator().manual_seed(0))

    @torch.no_grad()
    def reset_pool(self, gen):
        self.ab_c.zero_()
        self.c_norm.reset_parameters()
        if self.score_w is not None:
            self.score_w.normal_(0, 0.02, generator=gen)
        if self.gamma_raw is not None:
            hz = torch.logspace(math.log10(self.h_min), math.log10(self.h_max), self.h, dtype=torch.float64)
            gam = -torch.log1p(-1 / hz)
            self.gamma_raw.copy_(gam + torch.log(-torch.expm1(-gam)))
        self.c_proj.normal_(0, self.c_proj.shape[1] ** -0.5, generator=gen)

    def gamma(self):
        return None if self.gamma_raw is None else F.softplus(self.gamma_raw.float())

    def pooled(self, kv, cu):
        """kv (B,T,2*H*dk) = qkv[..., H*dk:] ([k heads ; v heads])."""
        if self.impl == "triton" and kv.is_cuda and not self.record_gate:
            B, T = kv.shape[:2]
            seg = segment_ids(B, T, cu, kv.device)
            if B > 1:
                seg = seg + torch.arange(B, device=kv.device)[:, None] * (T + 1)
            g = torch.zeros(self.h, device=kv.device) if self.gamma_raw is None else F.softplus(self.gamma_raw.float())
            dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.float32
            return pool_gate_triton(kv, self.c_proj, seg, self.score_w, g, self.c_norm.weight, self.bound,
                                    self.c_norm.eps, dt), None
        k, v = kv.unflatten(-1, (2, self.h, self.dk)).unbind(-3)
        seg = padded_segments(k.shape[0], k.shape[1], cu, self.chunk, k.device)
        fn = pool_gate_compiled if self.compiled else pool_gate
        args = (k, v, seg, self.c_proj, self.score_w, self.gamma_raw, self.c_norm.weight, self.chunk, self.bound,
                self.c_norm.eps)
        if self.record_gate:
            cn, s, c, g = pool_gate(*args, ret_c=True)
            self.last_gate.update(c=c.detach(), gamma=g.detach())
            return cn, s
        if self.ckpt and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x, cu=None, cu_cpu=None):
        qkv, _ = causal_conv1d(x=self.qkv(x), weight=self.conv, activation="silu", backend="triton", cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
        q, k, v = qkv.unflatten(-1, (3, self.h, self.dk)).unbind(-3)
        if self.record_gate:
            self.last_gate = {"c": None, "gamma": None}
        if self.mode == "span":
            return self.span(self, x, q, k, v, cu, cu_cpu)
        if not self.pool:
            a, b = self.ab(x).chunk(2, -1)
            s = None
        else:
            c, s = self.pooled(qkv[..., self.h * self.dk:], cu)
            a, b = (self.ab(x) + F.linear(c, self.ab_c)).chunk(2, -1)
        if self.record_gate:
            self.last_gate.update(a=a.detach(), b=b.detach(), s=None if s is None else s.detach())
        o, _ = chunk_gated_delta_rule(
            q, k, v, g=a, beta=b,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
        )
        o = self.o_norm(o, self.g_proj(x).unflatten(-1, (self.h, self.dk)))
        return self.o_proj(o.flatten(-2))


class SparseAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.hq, self.hkv, self.dh = c.nsa_q_heads, c.nsa_kv_heads, c.nsa_head_dim
        self.block_size, self.block_counts, self.window = c.nsa_block_size, c.nsa_block_counts, c.nsa_window
        self.q_proj = nn.Linear(c.d_model, self.hq * self.dh, bias=False)
        self.kv_proj = nn.Linear(c.d_model, 2 * self.hkv * self.dh, bias=False)
        self.gate_proj = nn.Linear(c.d_model, 3 * self.hq + self.hq * self.dh, bias=False)
        self.q_norm = RMSNorm(self.dh, eps=c.norm_eps)
        self.k_norm = RMSNorm(self.dh, eps=c.norm_eps)
        self.o_proj = nn.Linear(self.hq * self.dh, c.d_model, bias=False)

    def forward(self, x, cu=None, cu_cpu=None):
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.hq, self.dh)))
        k, v = self.kv_proj(x).unflatten(-1, (2, self.hkv, self.dh)).unbind(-3)
        k = self.k_norm(k)
        g, og = self.gate_proj(x).split([3 * self.hq, self.hq * self.dh], -1)
        g_cmp, g_slc, g_swa = g.unflatten(-1, (self.hq, 3)).sigmoid().unbind(-1)
        o = parallel_nsa(
            q, k, v, g_cmp=g_cmp, g_slc=g_slc,
            block_counts=self.block_counts,
            block_size=self.block_size,
            window_size=0,
            cu_seqlens=cu,
        )
        if self.window > 0:
            o = o + parallel_attn(q, k, v, window_size=self.window, cu_seqlens=cu) * g_swa.unsqueeze(-1)
        return self.o_proj(o.flatten(-2) * torch.sigmoid(og))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.RMSNorm(c.d_model, eps=c.norm_eps)
        self.w13 = nn.Linear(c.d_model, 2 * c.mlp_hidden, bias=False)
        self.w2 = nn.Linear(c.mlp_hidden, c.d_model, bias=False)

    def forward(self, x):
        a, b = self.w13(self.norm(x)).chunk(2, -1)
        return x + self.w2(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.norm = RMSNorm(c.d_model, eps=c.norm_eps)
        self.mixer = GatedDeltaNet(c) if kind == "G" else SparseAttention(c)
        self.mlp = MLP(c)

    def forward(self, x, cu=None, cu_cpu=None):
        return self.mlp(x + self.mixer(self.norm(x), cu, cu_cpu))


class LM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c, k) for k in c.kinds())
        self.norm = RMSNorm(c.d_model, eps=c.norm_eps)
        self.flce = FusedLinearCrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        std = self.c.init_std
        out_std = std / math.sqrt(2 * self.c.n_layers)
        nn.init.normal_(self.embed.weight, 0, std)
        for blk in self.blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, std)
            nn.init.normal_(blk.mixer.o_proj.weight, 0, out_std)
            nn.init.normal_(blk.mlp.w2.weight, 0, out_std)
        # pooled-gate extras use their own generator so the global RNG stream (and all shared params) match arm A
        for li, blk in enumerate(self.blocks):
            if getattr(blk.mixer, "pool", False):
                blk.mixer.reset_pool(torch.Generator().manual_seed(torch.initial_seed() + 7919 * (li + 1)))
            if getattr(blk.mixer, "mode", None) == "span":
                blk.mixer.span.reset(torch.Generator().manual_seed(torch.initial_seed() + 7919 * (li + 1)))

    def param_groups(self, weight_decay):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if p.ndim == 2 and not n.endswith((".conv", ".score_w")) else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def compile_mlps(self):
        for blk in self.blocks:
            blk.mlp.compile()
            if getattr(blk.mixer, "pool", False) and blk.mixer.compile_pool:
                blk.mixer.compiled = True

    def forward(self, idx, targets=None, cu=None, cu_cpu=None):
        if cu is not None:
            idx = idx.reshape(1, -1)
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, cu, cu_cpu)
        x = self.norm(x)
        if targets is None:
            return F.linear(x, self.embed.weight)
        targets = targets.reshape(x.shape[:2])
        if self.c.loss == "fused":
            return self.flce(x.to(torch.bfloat16), targets, self.embed.weight.to(torch.bfloat16))
        logits = F.linear(x, self.embed.weight).float()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
