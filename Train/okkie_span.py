# okkie-attention, span version (gate_mode = "span"). Used by model.GatedDeltaNet: the q, k, v path, conv, ab,
# A_log, dt_bias, o_norm, g_proj, o_proj are the parent's; this module owns the new parameters only (raw
# nn.Parameter, no nn.Linear: creating / initialising them never touches the global RNG, so shared params match
# gate_mode = "linear" for the same seed). Per block:
#   z_t = MLP(x_t) (H*span_dim), c_{t,h} = head piece.
#   spans: s_t = cos(EMA_short(z), EMA_long(z)); new span at t when s_t < theta <= s_{t-1}, or span_max_len tokens
#         after the last start; ids clamped to span_max-1.
#   mung_{j,h} = mean_{i in span j} P_h [k_i; v_i] (P_h per head, (mung_dim, 2*Dh)); zbar_j = mean z over span j.
#   read: usable spans = finished before t's span; sim = cos(z_t, zbar_j); top span_top_n usable -> restore
#         (shared Linear mung -> span_m pairs (k~, v~)); other usable -> coarse (shared Linear mung -> one pair);
#         softmax over scale_h * <l2(q), l2(k~)> plus a null slot (score 0, value 0); o_read = sum p v~.
#   lifetime gate: w = sigmoid(tau_h cos(c_{t,h}, m_h) + b_h), m_h updated once per pool_chunk tokens;
#         g' = w g, beta' = w beta, head output (before o_norm) = w o_delta + rho_h o_read.
# Documents: all of the above runs in a per-document layout (rows = documents), so nothing crosses a boundary.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from span_triton import lifetime_gate_triton, span_read_triton, top_idx


class DocLayout:
    """Maps (B,T,...) mixer tensors to a per-document layout (N,L,...) and back.
    cu None: identity (rows are documents). Equal-length documents: reshape. Otherwise: gather into rows padded to
    the longest document; `valid` (N,L) marks real positions (pads are at the end of a row, after all real ones)."""

    def __init__(self, B, T, cu, cu_cpu, device):
        self.kind, self.valid = "id", None
        self.N, self.L = B, T
        if cu is None:
            return
        cc = (cu_cpu if cu_cpu is not None else cu.cpu()).long()
        lens = cc[1:] - cc[:-1]
        self.N, self.L = len(lens), int(lens.max())
        if bool((lens == lens[0]).all()):
            self.kind = "reshape"
            return
        self.kind = "gather"
        ar = torch.arange(self.L)
        valid = ar[None] < lens[:, None]
        idx = (cc[:-1, None] + ar[None]).clamp(max=int(cc[-1]) - 1)
        self.valid = valid.to(device, non_blocking=True)
        self.idx = idx.to(device, non_blocking=True)

    def to_doc(self, x):
        if self.kind == "id":
            return x
        if self.kind == "reshape":
            return x.reshape(self.N, self.L, *x.shape[2:])
        y = x[0][self.idx]
        return y * self.valid.view(*self.valid.shape, *([1] * (y.dim() - 2))).to(y.dtype)

    def from_doc(self, y):
        if self.kind == "id":
            return y
        if self.kind == "reshape":
            return y.reshape(1, self.N * self.L, *y.shape[2:])
        return y[self.valid][None]


def l2n(x, eps=1e-6):
    """Same normalisation as fla l2norm (x * rsqrt(sum x^2 + eps))."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def ema(z, h):
    """Unnormalised causal exponential sum sum_{i<=t} lam^(t-i) z_i, lam = 1 - 1/h. z (N,L,D) float32.
    (cos is scale invariant, so the normalisation of an average is not needed.)"""
    L = z.shape[1]
    t = torch.arange(L, device=z.device)
    d = (t[:, None] - t[None, :]).float()
    W = torch.exp(d.clamp(min=0) * math.log1p(-1.0 / h)).masked_fill(d < 0, 0.0)
    return W @ z


def ema_chunk(z, h, C=64):
    """Same as ema, chunked closed form: O(L C D) instead of O(L^2 D)."""
    N, L, D = z.shape
    ll = math.log1p(-1.0 / h)
    nc = -(-L // C)
    if nc * C != L:
        z = F.pad(z, (0, 0, 0, nc * C - L))
    i = torch.arange(C, device=z.device)
    d = (i[:, None] - i[None, :]).float()
    W = torch.exp(d.clamp(min=0) * ll).masked_fill(d < 0, 0.0)
    intra = W @ z.view(N, nc, C, D)
    j = torch.arange(nc, device=z.device)
    dd = (j[:, None] - 1 - j[None, :]).float()
    Wc = torch.exp(dd.clamp(min=0) * (C * ll)).masked_fill(dd < 0, 0.0)
    carry = Wc @ intra[:, :, -1]
    y = intra + torch.exp((i + 1).float() * ll)[:, None] * carry[:, :, None]
    return y.view(N, nc * C, D)[:, :L]


@torch.no_grad()
def starts_to_ids(start, max_len=None):
    """start (N,L) bool: threshold starts, column 0 False (position 0 always starts span 0). Unclamped ids (N,L)
    long. max_len > 0: a span is also forced to start max_len tokens after the last start, i.e. the gap from a
    threshold start p to the next one is cut into ceil(gap / max_len) spans. Causal: id_t uses start[:, :t+1]."""
    if not max_len:
        return start.long().cumsum(1)
    t = torch.arange(start.shape[1], device=start.device)
    p = torch.where(start, t, 0).cummax(1).values  # last threshold start <= t (0 if none)
    p_prev = torch.cat([p.new_zeros(p.shape[0], 1), p[:, :-1]], 1)  # last threshold start < t
    inc = torch.where(start & (t > 0), (t - p_prev + max_len - 1) // max_len, 0)  # spans of the closed gap
    return inc.cumsum(1) + (t - p) // max_len


@torch.no_grad()
def span_ids(z, h_short, h_long, theta, span_max, valid=None, max_len=None):
    """Hard causal span ids (N,L) long from z (N,L,D). Position 0 starts span 0; a span starts at t >= 1 when
    s_t < theta and s_{t-1} >= theta, or (max_len > 0) max_len tokens after the last start; ids >= span_max are
    clamped into span_max - 1."""
    with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
        z = z.float()
        if valid is not None:
            z = z * valid[..., None]
        s = F.cosine_similarity(ema_chunk(z, h_short), ema_chunk(z, h_long), dim=-1, eps=1e-8)
        start = (s[:, 1:] < theta) & (s[:, :-1] >= theta)
        start = torch.cat([start.new_zeros(start.shape[0], 1), start], 1)
        return starts_to_ids(start, max_len).clamp(max=span_max - 1), s


def read_masks(sid, sim, top_n, S):
    """usable (N,L,S): span finished before t (id < sid_t). high: top_n usable spans by sim. low = usable & ~high."""
    usable = torch.arange(S, device=sid.device) < sid[..., None]
    top = sim.masked_fill(~usable, float("-inf")).topk(min(top_n, S), -1).indices
    high = torch.zeros_like(usable).scatter_(-1, top, True) & usable
    return high, usable & ~high


def span_read(q, mung, high, low, restore_w, coarse_w, log_scale, m):
    """q (N,L,H,Dh) f32; mung (N,S,H,R) f32; high/low (N,L,S) bool. Restore is computed for all spans and masked
    (equal to computing it for the selected ones only: spans do not interact before the softmax mask).
    Returns o_read (N,L,H,Dh) f32. A token with no usable span puts all weight on the null slot -> exactly 0."""
    N, S, H, R = mung.shape
    Dh = q.shape[-1]
    kr, vr = (mung @ restore_w.t()).view(N, S, H, m, 2, Dh).unbind(-2)
    kc, vc = (mung @ coarse_w.t()).view(N, S, H, 2, Dh).unbind(-2)
    qn = l2n(q) * log_scale.exp()[:, None]
    sh = torch.einsum("nlhd,nshmd->nlhsm", qn, l2n(kr)).masked_fill(~high[:, :, None, :, None], float("-inf"))
    sl = torch.einsum("nlhd,nshd->nlhs", qn, l2n(kc)).masked_fill(~low[:, :, None, :], float("-inf"))
    logits = torch.cat([sh.flatten(-2), sl, sl.new_zeros(*sl.shape[:-1], 1)], -1)
    p = torch.softmax(logits, -1)
    ph, pl = p[..., :S * m].unflatten(-1, (S, m)), p[..., S * m:S * m + S]
    return torch.einsum("nlhsm,nshmd->nlhd", ph, vr) + torch.einsum("nlhs,nshd->nlhd", pl, vc)


def lifetime_gate(c, g, beta, tau, bias, chunk):
    """c (N,L,H,P), g, beta (N,L,H) f32. Per chunk: tokens use m_h from before the chunk (0 -> cos 0); after it
    num <- exp(sum g') num + sum w beta c. Returns w (N,L,H)."""
    N, L, H, P = c.shape
    num = c.new_zeros(N, H, P)
    ws = []
    for s0 in range(0, L, chunk):
        cc = c[:, s0:s0 + chunk]
        cos = F.cosine_similarity(cc, num[:, None], dim=-1, eps=1e-8)
        w = torch.sigmoid(tau * cos + bias)
        ws.append(w)
        num = torch.exp((w * g[:, s0:s0 + chunk]).sum(1))[..., None] * num + \
            torch.einsum("nch,nchp->nhp", w * beta[:, s0:s0 + chunk], cc)
    return torch.cat(ws, 1)


class SpanSum(torch.autograd.Function):
    """Sums of k and v over each span (oh (N,L,S) 0/1 bf16; k, v bf16 in, f32 out, no f32 copy of k / v).
    Returns (N,S,H*Dh) x 2 f32; backward gathers the span gradient back to each token."""

    @staticmethod
    def forward(ctx, oh, sid, k, v):
        N, L, H, D = k.shape
        sk = torch.bmm(oh.transpose(1, 2), k.flatten(2), out_dtype=torch.float32)
        sv = torch.bmm(oh.transpose(1, 2), v.flatten(2), out_dtype=torch.float32)
        ctx.save_for_backward(sid)
        ctx.shape, ctx.dt = (N, L, H, D), k.dtype
        return sk, sv

    @staticmethod
    def backward(ctx, dk, dv):
        (sid,) = ctx.saved_tensors
        N, L, H, D = ctx.shape
        idx = sid[..., None].expand(N, L, H * D)
        return (None, None, torch.gather(dk, 1, idx).to(ctx.dt).view(N, L, H, D),
                torch.gather(dv, 1, idx).to(ctx.dt).view(N, L, H, D))


def combine(o, w, rho, o_read):
    """Head output before o_norm: w o_delta + rho_h o_read."""
    return (o.float() * w[..., None] + rho.float()[:, None] * o_read).to(o.dtype)


combine_compiled = torch.compile(combine, dynamic=False)


class OkkieSpan(nn.Module):
    def __init__(self, c, H, Dh):
        super().__init__()
        self.H, self.Dh = H, Dh
        self.hidden, self.P, self.R = c.span_hidden, c.span_dim, c.mung_dim
        self.h_short, self.h_long, self.theta = c.span_h_short, c.span_h_long, c.span_threshold
        self.S, self.top_n, self.m, self.chunk = c.span_max, c.span_top_n, c.span_m, c.pool_chunk
        self.ckpt = c.pool_ckpt
        self.max_len = c.span_max_len
        self.z_w1 = nn.Parameter(torch.empty(self.hidden, c.d_model))
        self.z_w2 = nn.Parameter(torch.empty(H * self.P, self.hidden))
        self.mung_w = nn.Parameter(torch.empty(H * self.R, 2 * Dh))  # per head: P_h = mung_w.view(H, R, 2Dh)[h]
        self.restore_w = nn.Parameter(torch.empty(self.m * 2 * Dh, self.R))  # shared across heads
        self.coarse_w = nn.Parameter(torch.empty(2 * Dh, self.R))  # shared across heads
        self.read_log_scale = nn.Parameter(torch.empty(H))
        self.tau = nn.Parameter(torch.empty(H))
        self.gate_b = nn.Parameter(torch.empty(H))
        self.rho = nn.Parameter(torch.empty(H))
        self.record = False
        self.last = None
        self.reset(torch.Generator().manual_seed(0))

    @torch.no_grad()
    def reset(self, gen):
        for w in (self.z_w1, self.z_w2, self.mung_w, self.restore_w, self.coarse_w):
            w.normal_(0, w.shape[1] ** -0.5, generator=gen)
        self.read_log_scale.fill_(0.5 * math.log(self.Dh))
        self.tau.zero_()
        self.gate_b.fill_(6.0)
        self.rho.zero_()

    def features(self, x):
        return F.linear(F.silu(F.linear(x, self.z_w1)), self.z_w2)

    def forward(self, p, x, q, k, v, cu=None, cu_cpu=None):
        """p: the parent GatedDeltaNet. x (B,T,d); q, k, v (B,T,H,Dh) after conv. Returns the mixer output."""
        B, T = x.shape[:2]
        H, Dh = self.H, self.Dh
        lay = DocLayout(B, T, cu, cu_cpu, x.device)
        z = self.features(x)
        a, b = p.ab(x).chunk(2, -1)
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            zf = z.float()
            g = -torch.exp(p.A_log.float()) * F.softplus(a.float() + p.dt_bias.float())
            beta = torch.sigmoid(b.float())
            zd = lay.to_doc(zf)
            N, L = zd.shape[:2]
            sid, _ = span_ids(zd, self.h_short, self.h_long, self.theta, self.S, lay.valid, self.max_len)
            w = lay.from_doc(lifetime_gate_triton(zd.view(N, L, H, self.P), lay.to_doc(g), lay.to_doc(beta),
                                                  self.tau.float(), self.gate_b.float(), self.chunk))
            S = int(sid.max()) + 1  # host sync: only spans 0..S-1 occur in this batch
            high = low = None
            if S == 1:
                # no usable span anywhere: o_read = 0 exactly; read params get exact zero grads
                zero = (self.restore_w.sum() + self.coarse_w.sum() + self.read_log_scale.sum() +
                        self.mung_w.sum()) * 0.0
                o_read = zero.expand(N, L, H, Dh)
                if self.record:
                    high = low = torch.zeros(N, L, 1, dtype=torch.bool, device=x.device)
            else:
                # spans: means as one-hot matmuls, mung = P_h applied to the mean of [k; v]
                oh = F.one_hot(sid, S)
                if lay.valid is not None:
                    oh = oh * lay.valid[..., None]
                cnt = oh.sum(1).float().clamp(min=1)  # (N,S)
                ohn = oh.float() / cnt[:, None]
                if lay.kind == "id" and k.dtype == torch.bfloat16:
                    sk, sv = SpanSum.apply(oh.to(torch.bfloat16), sid, k, v)
                    kvm = torch.stack([(sk / cnt[..., None]).view(N, S, H, Dh),
                                       (sv / cnt[..., None]).view(N, S, H, Dh)], 3).flatten(-2)
                else:
                    kvm = torch.einsum("nls,nlhd->nshd", ohn, lay.to_doc(torch.cat([k, v], -1).float()))
                mung = torch.einsum("nshd,hrd->nshr", kvm, self.mung_w.float().view(H, self.R, 2 * Dh))
                with torch.no_grad():
                    zbar = torch.einsum("nls,nld->nsd", ohn, zd)
                    sim = torch.einsum("nld,nsd->nls", F.normalize(zd, dim=-1, eps=1e-8),
                                       F.normalize(zbar, dim=-1, eps=1e-8))
                    if self.record:
                        high, low = read_masks(sid, sim, self.top_n, S)
                    top = top_idx(sid, sim, self.top_n, S)
                o_read = span_read_triton(lay.to_doc(q), mung, sid, top, self.restore_w.float(),
                                          self.coarse_w.float(), self.read_log_scale.float(), self.m)
            o_read = lay.from_doc(o_read)
        if p.record_gate:
            p.last_gate.update(a=a.detach(), b=b.detach(), s=None, w=w.detach())
        if self.record:
            self.last = {"sid": sid.detach(), "valid": lay.valid, "w": w.detach(), "high": high, "low": low,
                         "o_read": o_read.detach()}
        o, _ = chunk_gated_delta_rule(
            q, k, v, g=w * g, beta=w * beta,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            use_gate_in_kernel=False,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
        )
        o = combine_compiled(o, w, self.rho, o_read)
        o = p.o_norm(o, p.g_proj(x).unflatten(-1, (H, Dh)))
        return p.o_proj(o.flatten(-2))
