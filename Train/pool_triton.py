"""Fused Triton kernels for the pooled gate: score + causal recency-weighted softmax pooling + per-head RMSNorm.

Same maths as model.pool_gate (torch reference path):
  z_i = [k_i ; v_i] (per head, 2*DK dims), x_i = A_h z_i (R dims), s_i = bound * tanh(w_h . z_i / bound) (or 0),
  c_t = sum_{i<=t, seg_i == seg_t} exp(s_i - gamma_h (t-i)) x_i / sum_{...} exp(s_i - gamma_h (t-i)),
  out_t = nw * c_t / sqrt(mean(c_t^2) + eps).
The projection x = A z is one matmul in torch (pool_gate_triton, block-diagonal weight over all heads, in out_dtype;
its backward is torch autograd). The kernels (pool_core_triton) take x and k, v (for the score only): the score is
computed in float32 from k, v once (P1) and stored (N, H); all pooling arithmetic is float32 (dots with
input_precision="ieee"); x, k, v may be bf16 (k, v strided views into the qkv tensor).

Chunked over BT tokens. Forward: P1 (parallel over chunks) score + chunk-end sums E_n; P2 (sequential over chunks,
vector state) carry S_n = state at the end of chunk n-1; P3 (parallel) intra-chunk BT x BT weights + carry, divide,
norm. Backward: B1 (parallel) F_n = sum_t f_t dnum_t; B2 (reverse scan) dS; B3 (parallel) dx, ds -> dk, dv (score
path), dw, dgamma, dnw. Nothing of size (T, R) is stored for backward besides x: num/den are recomputed from x, s
and the per-chunk carry S.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _load_x(X, SEG, n, h, N, H: tl.constexpr, BT: tl.constexpr, R: tl.constexpr):
    t = n * BT + tl.arange(0, BT)
    m = t < N
    r = tl.arange(0, R)
    x = tl.load(X + (t[:, None] * H + h) * R + r[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    seg = tl.load(SEG + t, mask=m, other=-1)
    return t, m, x, seg


@triton.jit
def _score(K, V, Wt, t, m, h, stride_t, stride_h, bound, DK: tl.constexpr):
    d = tl.arange(0, DK)
    off = t[:, None] * stride_t + h * stride_h + d[None, :]
    zk = tl.load(K + off, mask=m[:, None], other=0.0).to(tl.float32)
    zv = tl.load(V + off, mask=m[:, None], other=0.0).to(tl.float32)
    wk = tl.load(Wt + h * 2 * DK + d)
    wv = tl.load(Wt + h * 2 * DK + DK + d)
    raw = tl.sum(zk * wk[None, :], 1) + tl.sum(zv * wv[None, :], 1)
    xs = raw / bound
    # tanh via exp of a non-positive argument (no overflow)
    e = tl.exp(-2.0 * tl.abs(xs))
    th = (1.0 - e) / (1.0 + e)
    th = tl.where(xs < 0, -th, th)
    return bound * th, th, zk, zv, wk, wv


@triton.jit
def _p1_chunk_end(K, V, X, SEG, Wt, G, SC, E, DEC, N, stride_t, stride_h, bound,
                  H: tl.constexpr, BT: tl.constexpr, DK: tl.constexpr, R: tl.constexpr, DP: tl.constexpr,
                  HAS_SCORE: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    t, m, x, seg = _load_x(X, SEG, n, h, N, H, BT, R)
    if HAS_SCORE:
        s, th, zk, zv, wk, wv = _score(K, V, Wt, t, m, h, stride_t, stride_h, bound, DK)
        tl.store(SC + t * H + h, s, mask=m)
    else:
        s = tl.zeros([BT], dtype=tl.float32)
    g = tl.load(G + h)
    last = tl.minimum(n * BT + BT - 1, N - 1)
    seg_last = tl.load(SEG + last)
    seg_prev_last = tl.load(SEG + tl.maximum(n * BT - 1, 0))
    tl_ = tl.arange(0, BT)
    wgt = tl.where(m & (seg == seg_last), tl.exp(s - g * (BT - 1 - tl_).to(tl.float32)), 0.0)
    r = tl.arange(0, R)
    base = E + (n * H + h) * DP
    tl.store(base + r, tl.sum(wgt[:, None] * x, 0))
    tl.store(base + R, tl.sum(wgt, 0))
    dec = tl.where((n > 0) & (seg_last == seg_prev_last), tl.exp(-g * BT), 0.0)
    tl.store(DEC + n * H + h, dec)


@triton.jit
def _scan(X, DEC, OUT, NC, H: tl.constexpr, DP: tl.constexpr, BD: tl.constexpr, REVERSE: tl.constexpr):
    """Forward: OUT[n] = cur; cur = X[n] + DEC[n] * cur  (cur_0 = 0).
    Reverse:  OUT[n] = cur; cur = X[n] + DEC[n] * cur  for n = NC-1 .. 0."""
    h = tl.program_id(0)
    j = tl.program_id(1) * BD + tl.arange(0, BD)
    mj = j < DP
    cur = tl.zeros([BD], dtype=tl.float32)
    for i in range(NC):
        if REVERSE:
            n = NC - 1 - i
        else:
            n = i
        off = (n * H + h) * DP + j
        x = tl.load(X + off, mask=mj, other=0.0)
        dec = tl.load(DEC + n * H + h)
        tl.store(OUT + off, cur, mask=mj)
        cur = x + dec * cur


@triton.jit
def _recompute(X, SEG, SC, G, S, NW, n, h, N, eps,
               H: tl.constexpr, BT: tl.constexpr, R: tl.constexpr, DP: tl.constexpr, HAS_SCORE: tl.constexpr):
    t, m, x, seg = _load_x(X, SEG, n, h, N, H, BT, R)
    if HAS_SCORE:
        s = tl.load(SC + t * H + h, mask=m, other=0.0)
    else:
        s = tl.zeros([BT], dtype=tl.float32)
    g = tl.load(G + h)
    tl_ = tl.arange(0, BT)
    dist = (tl_[:, None] - tl_[None, :]).to(tl.float32)
    mask = (dist >= 0) & (seg[:, None] == seg[None, :])
    W = tl.exp(tl.where(mask, s[None, :] - g * dist, float("-inf")))
    seg_prev = tl.where(n > 0, tl.load(SEG + tl.maximum(n * BT - 1, 0)), -2)
    f = tl.where(seg == seg_prev, tl.exp(-g * (tl_ + 1).to(tl.float32)), 0.0)
    r = tl.arange(0, R)
    base = S + (n * H + h) * DP
    sx = tl.load(base + r)
    sd = tl.load(base + R)
    num = tl.dot(W, x, input_precision="ieee") + f[:, None] * sx[None, :]
    den = tl.sum(W, 1) + f * sd
    den = tl.where(m, den, 1.0)
    c = num / den[:, None]
    ms = tl.sum(c * c, 1) / R
    rr = 1.0 / tl.sqrt(ms + eps)
    nw = tl.load(NW + r)
    return t, m, x, g, dist, W, f, sx, sd, den, c, rr, nw


@triton.jit
def _p3_out(X, SEG, SC, G, S, NW, OUT, N, eps,
            H: tl.constexpr, BT: tl.constexpr, R: tl.constexpr, DP: tl.constexpr, HAS_SCORE: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    t, m, x, g, dist, W, f, sx, sd, den, c, rr, nw = _recompute(X, SEG, SC, G, S, NW, n, h, N, eps,
                                                                 H, BT, R, DP, HAS_SCORE)
    r = tl.arange(0, R)
    off = (t[:, None] * H + h) * R + r[None, :]
    tl.store(OUT + off, (c * rr[:, None] * nw[None, :]).to(OUT.dtype.element_ty), mask=m[:, None])


@triton.jit
def _grad_num(DO, t, m, h, c, rr, den, nw, H: tl.constexpr, R: tl.constexpr):
    r = tl.arange(0, R)
    off = (t[:, None] * H + h) * R + r[None, :]
    do = tl.load(DO + off, mask=m[:, None], other=0.0).to(tl.float32)
    gg = do * nw[None, :]
    cg = tl.sum(c * gg, 1) / R
    r3 = rr * rr * rr
    dc = rr[:, None] * gg - c * (r3 * cg)[:, None]
    dnum = dc / den[:, None]
    dden = -tl.sum(dc * c, 1) / den
    return do, dnum, dden


@triton.jit
def _b1(X, SEG, SC, G, S, NW, DO, FX, N, eps,
        H: tl.constexpr, BT: tl.constexpr, R: tl.constexpr, DP: tl.constexpr, HAS_SCORE: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    t, m, x, g, dist, W, f, sx, sd, den, c, rr, nw = _recompute(X, SEG, SC, G, S, NW, n, h, N, eps,
                                                                 H, BT, R, DP, HAS_SCORE)
    do, dnum, dden = _grad_num(DO, t, m, h, c, rr, den, nw, H, R)
    r = tl.arange(0, R)
    base = FX + (n * H + h) * DP
    tl.store(base + r, tl.sum(f[:, None] * dnum, 0))
    tl.store(base + R, tl.sum(f * dden, 0))


@triton.jit
def _b3(X, SEG, SC, G, S, NW, DO, DSN, K, V, Wt, DX, DK_, DV_, DW, DG, DNW, N, stride_t, stride_h, bound, eps,
        H: tl.constexpr, BT: tl.constexpr, DK: tl.constexpr, R: tl.constexpr, DP: tl.constexpr,
        HAS_SCORE: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    t, m, x, g, dist, W, f, sx, sd, den, c, rr, nw = _recompute(X, SEG, SC, G, S, NW, n, h, N, eps,
                                                                 H, BT, R, DP, HAS_SCORE)
    do, dnum, dden = _grad_num(DO, t, m, h, c, rr, den, nw, H, R)
    r = tl.arange(0, R)
    # dnw partial
    pb = (n * H + h) * R
    tl.store(DNW + pb + r, tl.sum(do * c * rr[:, None], 0))
    # add the carry gradient dS_{n+1} on the last row (num at the chunk end is the next chunk's carry)
    base = DSN + (n * H + h) * DP
    tl_ = tl.arange(0, BT)
    is_last = (tl_ == BT - 1).to(tl.float32)
    Gx = dnum + is_last[:, None] * tl.load(base + r)[None, :]
    Gd = dden + is_last * tl.load(base + R)
    Gx = tl.where(m[:, None], Gx, 0.0)
    Gd = tl.where(m, Gd, 0.0)
    dx = tl.dot(tl.trans(W), Gx, input_precision="ieee")
    dW = tl.dot(Gx, tl.trans(x), input_precision="ieee") + Gd[:, None]
    dE = dW * W
    ds = tl.sum(dE, 0)
    df = tl.sum(Gx * sx[None, :], 1) + Gd * sd
    dg = -tl.sum(tl.sum(dist * dE, 1), 0) - tl.sum((tl_ + 1).to(tl.float32) * f * df, 0)
    tl.store(DG + n * H + h, dg)
    off = (t[:, None] * H + h) * R + r[None, :]
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=m[:, None])
    if HAS_SCORE:
        s, th, zk, zv, wk, wv = _score(K, V, Wt, t, m, h, stride_t, stride_h, bound, DK)
        draw = ds * (1.0 - th * th)
        draw = tl.where(m, draw, 0.0)
        d = tl.arange(0, DK)
        pw = (n * H + h) * (2 * DK)
        tl.store(DW + pw + d, tl.sum(draw[:, None] * zk, 0))
        tl.store(DW + pw + DK + d, tl.sum(draw[:, None] * zv, 0))
        offk = (t[:, None] * H + h) * DK + d[None, :]
        tl.store(DK_ + offk, (draw[:, None] * wk[None, :]).to(DK_.dtype.element_ty), mask=m[:, None])
        tl.store(DV_ + offk, (draw[:, None] * wv[None, :]).to(DV_.dtype.element_ty), mask=m[:, None])


BT_DEFAULT = 32
NUM_WARPS = 8
BD_SCAN = 64


class PoolGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, x, w, gamma, nw, seg, bound, eps, out_dtype, bt):
        B, T, H, R = x.shape
        assert R >= 16 and R & (R - 1) == 0, f"pool rank {R} must be a power of 2 >= 16 (tl.dot)"
        N = B * T
        x2 = x.reshape(N, H, R).contiguous()
        DP = R + 1
        nc = triton.cdiv(N, bt)
        dev = x.device
        has_score = w is not None
        if has_score:
            DK = k.shape[-1]
            k2, v2 = k.reshape(N, H, DK), v.reshape(N, H, DK)
            assert k2.stride() == v2.stride() and k2.stride(-1) == 1, (k2.stride(), v2.stride())
            st, sh = k2.stride(0), k2.stride(1)
            wt = w.float().contiguous()
            SC = torch.empty(N, H, device=dev)
        else:
            DK, k2, v2, st, sh = 16, x2, x2, 0, 0
            wt = SC = torch.zeros(1, device=dev)
        g = gamma.float().contiguous()
        nwf = nw.float().contiguous()
        seg = seg.reshape(N).to(torch.int32).contiguous()
        E = torch.empty(nc, H, DP, device=dev)
        DEC = torch.empty(nc, H, device=dev)
        S = torch.empty(nc, H, DP, device=dev)
        out = torch.empty(N, H, R, device=dev, dtype=out_dtype)
        _p1_chunk_end[(nc, H)](k2, v2, x2, seg, wt, g, SC, E, DEC, N, st, sh, bound, H=H, BT=bt, DK=DK, R=R, DP=DP,
                               HAS_SCORE=has_score, num_warps=NUM_WARPS)
        _scan[(H, triton.cdiv(DP, BD_SCAN))](E, DEC, S, nc, H=H, DP=DP, BD=BD_SCAN, REVERSE=False, num_warps=1)
        _p3_out[(nc, H)](x2, seg, SC, g, S, nwf, out, N, eps, H=H, BT=bt, R=R, DP=DP,
                         HAS_SCORE=has_score, num_warps=NUM_WARPS)
        ctx.save_for_backward(k, v, x2, wt, g, nwf, seg, SC, S, DEC)
        ctx.cfg = (bound, eps, bt, has_score, x.shape)
        return out.view(B, T, H * R)

    @staticmethod
    def backward(ctx, dout):
        k, v, x2, wt, g, nwf, seg, SC, S, DEC = ctx.saved_tensors
        bound, eps, bt, has_score, xshape = ctx.cfg
        N, H, R = x2.shape
        DP = R + 1
        nc = triton.cdiv(N, bt)
        dev = x2.device
        do = dout.reshape(N, H, R).contiguous()
        FX = torch.empty(nc, H, DP, device=dev)
        DSN = torch.empty(nc, H, DP, device=dev)
        _b1[(nc, H)](x2, seg, SC, g, S, nwf, do, FX, N, eps, H=H, BT=bt, R=R, DP=DP,
                     HAS_SCORE=has_score, num_warps=NUM_WARPS)
        _scan[(H, triton.cdiv(DP, BD_SCAN))](FX, DEC, DSN, nc, H=H, DP=DP, BD=BD_SCAN, REVERSE=True, num_warps=1)
        dx = torch.empty(N, H, R, device=dev, dtype=x2.dtype)
        if has_score:
            DK = k.shape[-1]
            k2, v2 = k.reshape(N, H, DK), v.reshape(N, H, DK)
            st, sh = k2.stride(0), k2.stride(1)
            dk = torch.empty(N, H, DK, device=dev, dtype=k.dtype)
            dv = torch.empty(N, H, DK, device=dev, dtype=v.dtype)
            DWp = torch.empty(nc, H, 2 * DK, device=dev)
        else:
            DK, k2, v2, st, sh = 16, x2, x2, 0, 0
            dk = dv = DWp = torch.zeros(1, device=dev)
        DGp = torch.empty(nc, H, device=dev)
        DNWp = torch.empty(nc, H, R, device=dev)
        _b3[(nc, H)](x2, seg, SC, g, S, nwf, do, DSN, k2, v2, wt, dx, dk, dv, DWp, DGp, DNWp, N, st, sh, bound, eps,
                     H=H, BT=bt, DK=DK, R=R, DP=DP, HAS_SCORE=has_score, num_warps=NUM_WARPS)
        if has_score:
            dk, dv, dw = dk.view(k.shape), dv.view(v.shape), DWp.sum(0)
        else:
            dk = dv = dw = None
        return dk, dv, dx.view(xshape), dw, DGp.sum(0), DNWp.sum((0, 1)), None, None, None, None, None


def pool_core_triton(k, v, x, seg, score_w, gamma, norm_w, bound, eps, out_dtype=torch.float32, bt=BT_DEFAULT):
    """Pooling core on given x (B,T,H,R). k, v (B,T,H,DK) are read only for the score (unused if score_w is None);
    seg (B,T) int document ids, nondecreasing over the flattened B*T order; gamma (H,) float (>= 0).
    Returns RMSNorm_head(c) as (B,T,H*R) in out_dtype."""
    return PoolGate.apply(k, v, x, score_w, gamma, norm_w, seg, float(bound), float(eps), out_dtype, bt)


def project(kv, proj_w, H, dtype):
    """x = A_h [k; v] for all heads as one matmul. kv (B,T,2*H*DK) = [k (H,DK) ; v (H,DK)] flattened (the qkv[..., H*DK:]
    slice); proj_w (H*R, 2*DK). Block-diagonal weight (H*R, 2*H*DK); computed in dtype. Returns (B,T,H,R)."""
    R = proj_w.shape[0] // H
    DK = proj_w.shape[1] // 2
    A = proj_w.view(H, R, 2 * DK)
    Wb = torch.cat([torch.block_diag(*A[..., :DK]), torch.block_diag(*A[..., DK:])], 1)
    with torch.autocast("cuda", enabled=False):
        return F.linear(kv.to(dtype), Wb.to(dtype)).unflatten(-1, (H, R))


def pool_gate_triton(kv, proj_w, seg, score_w, gamma, norm_w, bound, eps, out_dtype=torch.float32, bt=BT_DEFAULT):
    """kv (B,T,2*H*DK) [k heads ; v heads]; proj_w (H*R, 2*DK); seg (B,T); gamma (H,) float (>= 0).
    x = proj_w-projection of [k; v] in out_dtype (torch matmul), then the fused pooling core.
    Returns RMSNorm_head(c) as (B,T,H*R) in out_dtype."""
    R = norm_w.shape[0]
    H = proj_w.shape[0] // R
    DK = proj_w.shape[1] // 2
    k, v = kv.unflatten(-1, (2, H, DK)).unbind(-3)
    x = project(kv, proj_w, H, out_dtype)
    return pool_core_triton(k, v, x, seg, score_w, gamma, norm_w, bound, eps, out_dtype, bt)
