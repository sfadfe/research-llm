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

    def kinds(self):
        return [self.pattern[i % len(self.pattern)] for i in range(self.n_layers)]


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

    def forward(self, x, cu=None, cu_cpu=None):
        qkv, _ = causal_conv1d(x=self.qkv(x), weight=self.conv, activation="silu", backend="triton", cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)
        q, k, v = qkv.unflatten(-1, (3, self.h, self.dk)).unbind(-3)
        a, b = self.ab(x).chunk(2, -1)
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

    def param_groups(self, weight_decay):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if p.ndim == 2 and not n.endswith(".conv") else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def compile_mlps(self):
        for blk in self.blocks:
            blk.mlp.compile()

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
