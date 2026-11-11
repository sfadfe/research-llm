"""Triton kernels for okkie_span: lifetime gate (fwd/bwd) and span read attention (fwd/bwd).
Both follow the torch references okkie_span.lifetime_gate / okkie_span.span_read (float32; the read's dots in
lowp=2 use 3x bf16 split products, rel error ~1e-5). top_idx / span_read_triton assume span_top_n <= 2."""
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------------------------------------ lifetime gate
@triton.jit
def _life_fwd(C, G, BT, TAU, BIAS, W, NUMS, L, H: tl.constexpr, P: tl.constexpr, CH: tl.constexpr, NCH,
              EPS: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    tau = tl.load(TAU + h)
    bias = tl.load(BIAS + h)
    ot = tl.arange(0, CH)
    op = tl.arange(0, P)
    num = tl.zeros([P], dtype=tl.float32)
    for j in range(NCH):
        t = j * CH + ot
        mt = t < L
        c = tl.load(C + ((n * L + t[:, None]) * H + h) * P + op[None, :], mask=mt[:, None], other=0.0)
        tl.store(NUMS + ((n * NCH + j) * H + h) * P + op, num)
        cn = tl.maximum(tl.sqrt(tl.sum(c * c, 1)), EPS)
        nn = tl.maximum(tl.sqrt(tl.sum(num * num, 0)), EPS)
        cos = tl.sum((c / cn[:, None]) * (num / nn)[None, :], 1)
        w = tl.sigmoid(tau * cos + bias)
        tl.store(W + (n * L + t) * H + h, w, mask=mt)
        g = tl.load(G + (n * L + t) * H + h, mask=mt, other=0.0)
        be = tl.load(BT + (n * L + t) * H + h, mask=mt, other=0.0)
        w = tl.where(mt, w, 0.0)
        e = tl.exp(tl.sum(w * g, 0))
        num = e * num + tl.sum((w * be)[:, None] * c, 0)


@triton.jit
def _life_bwd(C, G, BT, TAU, BIAS, NUMS, DW, DC, DG, DB, DTAU, DBIAS, L, H: tl.constexpr, P: tl.constexpr,
              CH: tl.constexpr, NCH, EPS: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // H
    h = pid % H
    tau = tl.load(TAU + h)
    bias = tl.load(BIAS + h)
    ot = tl.arange(0, CH)
    op = tl.arange(0, P)
    dnum = tl.zeros([P], dtype=tl.float32)
    dtau = tl.zeros([CH], dtype=tl.float32)
    dbias = tl.zeros([CH], dtype=tl.float32)
    for jj in range(NCH):
        j = NCH - 1 - jj
        t = j * CH + ot
        mt = t < L
        c = tl.load(C + ((n * L + t[:, None]) * H + h) * P + op[None, :], mask=mt[:, None], other=0.0)
        num = tl.load(NUMS + ((n * NCH + j) * H + h) * P + op)
        g = tl.load(G + (n * L + t) * H + h, mask=mt, other=0.0)
        be = tl.load(BT + (n * L + t) * H + h, mask=mt, other=0.0)
        dw = tl.load(DW + (n * L + t) * H + h, mask=mt, other=0.0)
        cn = tl.maximum(tl.sqrt(tl.sum(c * c, 1)), EPS)
        nn = tl.maximum(tl.sqrt(tl.sum(num * num, 0)), EPS)
        ch = c / cn[:, None]
        nh = num / nn
        cos = tl.sum(ch * nh[None, :], 1)
        w = tl.sigmoid(tau * cos + bias)
        w = tl.where(mt, w, 0.0)
        e = tl.exp(tl.sum(w * g, 0))
        # num_{j+1} = e num_j + sum_t (w beta)_t c_t
        dswg = tl.sum(dnum * num, 0) * e
        da = tl.sum(c * dnum[None, :], 1)
        a = w * be
        dc = a[:, None] * dnum[None, :]
        dwt = dw + dswg * g + da * be
        tl.store(DG + (n * L + t) * H + h, dswg * w, mask=mt)
        tl.store(DB + (n * L + t) * H + h, da * w, mask=mt)
        dpre = tl.where(mt, dwt * w * (1.0 - w), 0.0)
        dtau += dpre * cos
        dbias += dpre
        dcos = dpre * tau
        dc += dcos[:, None] * (nh[None, :] - ch * cos[:, None]) / cn[:, None]
        tl.store(DC + ((n * L + t[:, None]) * H + h) * P + op[None, :], dc, mask=mt[:, None])
        dnn = tl.sum(dcos[:, None] * (ch - nh[None, :] * cos[:, None]), 0) / nn
        dnum = e * dnum + tl.where(j > 0, dnn, 0.0)
    tl.store(DTAU + n * H + h, tl.sum(dtau, 0))
    tl.store(DBIAS + n * H + h, tl.sum(dbias, 0))


class LifetimeGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, g, beta, tau, bias, chunk):
        c, g, beta = c.contiguous(), g.contiguous(), beta.contiguous()
        N, L, H, P = c.shape
        nch = triton.cdiv(L, chunk)
        w = torch.empty(N, L, H, device=c.device, dtype=torch.float32)
        nums = torch.empty(N, nch, H, P, device=c.device, dtype=torch.float32)
        _life_fwd[(N * H,)](c, g, beta, tau, bias, w, nums, L, H, P, chunk, nch, 1e-8)
        ctx.save_for_backward(c, g, beta, tau, bias, nums)
        ctx.chunk = chunk
        return w

    @staticmethod
    def backward(ctx, dw):
        c, g, beta, tau, bias, nums = ctx.saved_tensors
        N, L, H, P = c.shape
        nch = nums.shape[1]
        dc, dg, db = torch.empty_like(c), torch.empty_like(g), torch.empty_like(beta)
        dtau = torch.empty(N, H, device=c.device, dtype=torch.float32)
        dbias = torch.empty_like(dtau)
        _life_bwd[(N * H,)](c, g, beta, tau, bias, nums, dw.contiguous(), dc, dg, db, dtau, dbias, L, H, P,
                            ctx.chunk, nch, 1e-8)
        return dc, dg, db, dtau.sum(0), dbias.sum(0), None


def lifetime_gate_triton(c, g, beta, tau, bias, chunk):
    return LifetimeGate.apply(c.float(), g.float(), beta.float(), tau.float(), bias.float(), chunk)


# ------------------------------------------------------------------------------------------------ span read
# keys per (n, h): KS = S * (m + 1) rows, span s owns rows s*(m+1) .. s*(m+1)+m: m restored keys, then the coarse key.
# token t may use span s iff s < sid_t; restored rows iff s in top_t, coarse row iff not. Null slot: logit 0, value 0.
@triton.jit
def _dot(a, b, LOWP: tl.constexpr):
    """a @ b. LOWP 0: fp32 (ieee FMA), 1: bf16 inputs fp32 acc, 2: 3x bf16 split (hi*hi + hi*lo + lo*hi)."""
    if LOWP == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif LOWP == 1:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        ah = a.to(tl.bfloat16)
        bh = b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        return tl.dot(ah, bh) + tl.dot(ah, bl) + tl.dot(al, bh)


@triton.jit
def _mask(qk, sid, t0, t1, j, M1: tl.constexpr, M: tl.constexpr):
    s = j // M1
    r = j - s * M1
    usable = s[None, :] < sid[:, None]
    sel = (s[None, :] == t0[:, None]) | (s[None, :] == t1[:, None])
    ok = usable & tl.where(r[None, :] < M, sel, ~sel)
    return tl.where(ok, qk, float("-inf"))


@triton.jit
def _qn(Q, SCALE, n, h, t, mt, L, H: tl.constexpr, D: tl.constexpr):
    od = tl.arange(0, D)
    q = tl.load(Q + ((n * L + t[:, None]) * H + h) * D + od[None, :], mask=mt[:, None], other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(q * q, 1) + 1e-6)
    return q, r, q * (r * tl.load(SCALE + h))[:, None]


@triton.jit
def _read_fwd(Q, SCALE, K, V, SID, TOP, O, LSE, L, KS, H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
              BL: tl.constexpr, BK: tl.constexpr, LOWP: tl.constexpr):
    tb = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid // H
    h = pid % H
    t = tb * BL + tl.arange(0, BL)
    mt = t < L
    od = tl.arange(0, D)
    _, _, qn = _qn(Q, SCALE, n, h, t, mt, L, H, D)
    sid = tl.load(SID + n * L + t, mask=mt, other=0)
    t0 = tl.load(TOP + (n * L + t) * 2, mask=mt, other=-1)
    t1 = tl.load(TOP + (n * L + t) * 2 + 1, mask=mt, other=-1)
    smax = tl.max(sid, 0)
    nk = tl.minimum(smax * (M + 1), KS)
    m_i = tl.zeros([BL], dtype=tl.float32)
    l_i = tl.full([BL], 1.0, dtype=tl.float32)
    acc = tl.zeros([BL, D], dtype=tl.float32)
    base = (n * H + h) * KS
    for k0 in range(0, nk, BK):
        j = k0 + tl.arange(0, BK)
        mk = j < KS
        k = tl.load(K + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
        v = tl.load(V + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
        qk = _dot(qn, tl.trans(k), LOWP)
        qk = _mask(qk, sid, t0, t1, j, M + 1, M)
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + _dot(p, v, LOWP)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + ((n * L + t[:, None]) * H + h) * D + od[None, :], o, mask=mt[:, None])
    tl.store(LSE + (n * H + h) * L + t, m_i + tl.log(l_i), mask=mt)


@triton.jit
def _read_bwd_dq(Q, SCALE, K, V, SID, TOP, LSE, DO, DELTA, DQ, DLS, L, KS, H: tl.constexpr, D: tl.constexpr,
                 M: tl.constexpr, BL: tl.constexpr, BK: tl.constexpr, LOWP: tl.constexpr):
    tb = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid // H
    h = pid % H
    t = tb * BL + tl.arange(0, BL)
    mt = t < L
    od = tl.arange(0, D)
    q, r, qn = _qn(Q, SCALE, n, h, t, mt, L, H, D)
    sid = tl.load(SID + n * L + t, mask=mt, other=0)
    t0 = tl.load(TOP + (n * L + t) * 2, mask=mt, other=-1)
    t1 = tl.load(TOP + (n * L + t) * 2 + 1, mask=mt, other=-1)
    lse = tl.load(LSE + (n * H + h) * L + t, mask=mt, other=0.0)
    do = tl.load(DO + ((n * L + t[:, None]) * H + h) * D + od[None, :], mask=mt[:, None], other=0.0)
    delta = tl.load(DELTA + (n * L + t) * H + h, mask=mt, other=0.0)
    smax = tl.max(sid, 0)
    nk = tl.minimum(smax * (M + 1), KS)
    dqn = tl.zeros([BL, D], dtype=tl.float32)
    base = (n * H + h) * KS
    for k0 in range(0, nk, BK):
        j = k0 + tl.arange(0, BK)
        mk = j < KS
        k = tl.load(K + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
        v = tl.load(V + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
        qk = _dot(qn, tl.trans(k), LOWP)
        dp = _dot(do, tl.trans(v), LOWP)
        qk = _mask(qk, sid, t0, t1, j, M + 1, M)
        p = tl.exp(qk - lse[:, None])
        ds = p * (dp - delta[:, None])
        dqn += _dot(ds, k, LOWP)
    sc = tl.load(SCALE + h)
    tl.store(DLS + (n * H + h) * tl.num_programs(0) + tb, tl.sum(tl.sum(dqn * qn, 1), 0))
    dqh = dqn * sc
    dq = r[:, None] * dqh - q * (r * r * r * tl.sum(q * dqh, 1))[:, None]
    tl.store(DQ + ((n * L + t[:, None]) * H + h) * D + od[None, :], dq.to(DQ.dtype.element_ty), mask=mt[:, None])


@triton.jit
def _read_bwd_dkv(Q, SCALE, K, V, SID, TOP, LSE, DO, DELTA, DK, DV, L, KS, H: tl.constexpr, D: tl.constexpr,
                  M: tl.constexpr, BL: tl.constexpr, BK: tl.constexpr, LOWP: tl.constexpr):
    kb = tl.program_id(0)
    pid = tl.program_id(1)
    n = pid // H
    h = pid % H
    od = tl.arange(0, D)
    j = kb * BK + tl.arange(0, BK)
    mk = j < KS
    base = (n * H + h) * KS
    k = tl.load(K + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
    v = tl.load(V + (base + j[:, None]) * D + od[None, :], mask=mk[:, None], other=0.0)
    s_lo = (kb * BK) // (M + 1)
    dk = tl.zeros([BK, D], dtype=tl.float32)
    dv = tl.zeros([BK, D], dtype=tl.float32)
    for t0_ in range(0, L, BL):
        last = tl.load(SID + n * L + tl.minimum(t0_ + BL, L) - 1)
        if last > s_lo:
            t = t0_ + tl.arange(0, BL)
            mt = t < L
            _, _, qn = _qn(Q, SCALE, n, h, t, mt, L, H, D)
            sid = tl.load(SID + n * L + t, mask=mt, other=0)
            ta = tl.load(TOP + (n * L + t) * 2, mask=mt, other=-1)
            tb_ = tl.load(TOP + (n * L + t) * 2 + 1, mask=mt, other=-1)
            lse = tl.load(LSE + (n * H + h) * L + t, mask=mt, other=0.0)
            do = tl.load(DO + ((n * L + t[:, None]) * H + h) * D + od[None, :], mask=mt[:, None], other=0.0)
            delta = tl.load(DELTA + (n * L + t) * H + h, mask=mt, other=0.0)
            qk = _dot(qn, tl.trans(k), LOWP)
            dp = _dot(do, tl.trans(v), LOWP)
            qk = _mask(qk, sid, ta, tb_, j, M + 1, M)
            p = tl.exp(qk - lse[:, None])
            ds = p * (dp - delta[:, None])
            dv += _dot(tl.trans(p), do, LOWP)
            dk += _dot(tl.trans(ds), qn, LOWP)
    tl.store(DK + (base + j[:, None]) * D + od[None, :], dk, mask=mk[:, None])
    tl.store(DV + (base + j[:, None]) * D + od[None, :], dv, mask=mk[:, None])


WARPS = [4, 4, 4]


class SpanRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, scale, kk, vv, sid, top, m, lowp, bl, bk):
        """q (N,L,H,D) any float dtype; scale (H,) = exp(log_scale); kk, vv (N,H,KS,D) f32 (kk l2-normalised);
        sid (N,L) int32; top (N,L,2) int32 (-1 = none). Returns o (N,L,H,D) f32."""
        q, kk, vv = q.contiguous(), kk.contiguous(), vv.contiguous()
        N, L, H, D = q.shape
        KS = kk.shape[2]
        o = torch.empty(N, L, H, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(N, H, L, device=q.device, dtype=torch.float32)
        _read_fwd[(triton.cdiv(L, bl), N * H)](q, scale, kk, vv, sid, top, o, lse, L, KS, H, D, m, bl, bk, lowp,
                                                  num_warps=WARPS[0])
        ctx.save_for_backward(q, scale, kk, vv, sid, top, o, lse)
        ctx.cfg = (m, lowp, bl, bk)
        return o

    @staticmethod
    def backward(ctx, do):
        q, scale, kk, vv, sid, top, o, lse = ctx.saved_tensors
        m, lowp, bl, bk = ctx.cfg
        N, L, H, D = q.shape
        KS = kk.shape[2]
        do = do.contiguous()
        delta = (do * o).sum(-1)
        nb = triton.cdiv(L, bl)
        dq = torch.empty_like(q)
        dls = torch.empty(N * H, nb, device=q.device, dtype=torch.float32)
        _read_bwd_dq[(nb, N * H)](q, scale, kk, vv, sid, top, lse, do, delta, dq, dls, L, KS, H, D, m, bl, bk, lowp,
                                     num_warps=WARPS[1])
        dk, dv = torch.empty_like(kk), torch.empty_like(vv)
        _read_bwd_dkv[(triton.cdiv(KS, bk), N * H)](q, scale, kk, vv, sid, top, lse, do, delta, dk, dv, L, KS, H, D,
                                                    m, bl, bk, lowp, num_warps=WARPS[2])
        # d log_scale = sum <dqn, qn>; scale = exp(log_scale) -> d scale = that / scale
        dscale = dls.view(N, H, nb).sum((0, 2)) / scale
        return dq, dscale, dk, dv, None, None, None, None, None, None


def l2n(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def top_idx(sid, sim, top_n, S):
    """(N,L,2) int32 span ids of the top_n usable spans by sim, -1 where fewer are usable (top_n <= 2)."""
    assert top_n <= 2, "span_read_triton handles span_top_n <= 2"
    usable = torch.arange(S, device=sid.device) < sid[..., None]
    top = sim.masked_fill(~usable, float("-inf")).topk(min(top_n, S), -1).indices
    top = torch.where(top < sid[..., None], top, -1)
    if top.shape[-1] < 2:
        top = torch.cat([top, top.new_full((*top.shape[:-1], 2 - top.shape[-1]), -1)], -1)
    return top.int().contiguous()


def span_read_triton(q, mung, sid, top, restore_w, coarse_w, log_scale, m, lowp=2, bl=64, bk=16):
    N, S, H, R = mung.shape
    Dh = q.shape[-1]
    kr, vr = (mung @ restore_w.t()).view(N, S, H, m, 2, Dh).unbind(-2)
    kc, vc = (mung @ coarse_w.t()).view(N, S, H, 2, Dh).unbind(-2)
    kk = torch.cat([l2n(kr), l2n(kc)[:, :, :, None]], 3).permute(0, 2, 1, 3, 4).reshape(N, H, S * (m + 1), Dh)
    vv = torch.cat([vr, vc[:, :, :, None]], 3).permute(0, 2, 1, 3, 4).reshape(N, H, S * (m + 1), Dh)
    return SpanRead.apply(q, log_scale.exp(), kk, vv, sid.int().contiguous(), top, m, lowp, bl, bk)
