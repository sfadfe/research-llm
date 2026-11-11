"""Post-run evaluation. Usage: python Train/evaluate.py <run dir> [--ckpt latest.pt] [--out results]
Writes results/<run>/{ppl,position_loss,gates,train}.json and summary.json (written last = done marker)."""
import argparse
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import LM, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
T = 2048
MB = 8
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
GATE_SEQS = 64  # sequences per source for gate statistics (64 * 2048 * 5 sources ~ 655k tokens)


def val_rows(name, eot):
    """Val sequences of T+1 tokens; doc_start[i] = row i begins at a true document start."""
    m = np.memmap(ROOT / "data/bin" / name / "val.bin", dtype=np.uint16, mode="r")
    k = (len(m) - 1) // T
    k -= k % MB
    starts = np.arange(k) * T
    rows = np.stack([m[o:o + T + 1] for o in starts]).astype(np.int64)
    doc_start = np.array([o == 0 or m[o - 1] == eot for o in starts])
    return rows, doc_start


def tensors(buf, eot, device):
    x, y = buf[:, :-1], buf[:, 1:]
    B, L = x.shape
    pos = np.flatnonzero(x.reshape(-1) == eot) + 1
    cu_cpu = torch.from_numpy(np.union1d(np.arange(0, B * L + 1, L), pos))
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return to(x), to(y), cu_cpu.to(device), cu_cpu


def positions(cu_cpu, n):
    """Position of each flattened token since its segment (doc-mask reset) start, and the segment start index."""
    cu = cu_cpu.numpy()
    seg = np.searchsorted(cu, np.arange(n), side="right") - 1
    return np.arange(n) - cu[seg], cu[seg]


def stats(a, lo, hi, bins=100):
    a = a.astype(np.float64)
    h, e = np.histogram(np.clip(a, lo, hi), bins=bins, range=(lo, hi))
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "quantiles": dict(zip(map(str, QS), map(float, np.quantile(a, QS)))),
            "hist": {"lo": lo, "hi": hi, "bins": bins, "counts": h.tolist()}}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="latest.pt")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    run = Path(args.run).resolve()
    out = Path(args.out) / run.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    cfg = tomllib.load(open(run / "config.toml", "rb"))
    meta = json.load(open(ROOT / "data/bin/meta.json"))
    eot = meta["eot"]
    sources = list(cfg["data"]["mix"])
    mc = ModelConfig(**cfg["model"])
    model = LM(mc).to(device).eval()
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    step = ck["step"]
    del ck
    gdn = [blk.mixer for blk in model.blocks if blk.kind == "G"]

    # buckets for position-in-document loss: [0,1), [1,2), [2,4), ..., [1024,2048)
    edges = [0] + [2 ** i for i in range(12)]
    nb = len(edges)
    ppl, pos_res = {}, {}
    gate_acc = {"alpha": [[] for _ in gdn], "beta": [[] for _ in gdn], "score": [[] for _ in gdn]}
    for name in sources:
        rows, doc_start = val_rows(name, eot)
        s_loss = n_tok = 0.0
        pb = {k: np.zeros((2, nb)) for k in ("doc", "context")}
        for i in range(0, len(rows), MB):
            buf = rows[i:i + MB]
            x, y, cu, cu_cpu = tensors(buf, eot, device)
            rec = i < GATE_SEQS
            for m in gdn:
                m.rec = {} if rec else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, None, cu, cu_cpu)
            tok = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none").cpu().numpy()
            del logits
            s_loss += float(tok.astype(np.float64).sum())
            n_tok += tok.size
            p, st = positions(cu_cpu, tok.size)
            b = np.searchsorted(edges, p, side="right") - 1
            # a segment is a true document iff it starts after an EOT, or at a row start that is a doc start
            row_start = st % T == 0
            true_doc = ~row_start | doc_start[i:i + MB][st // T]
            for key, sel in (("context", np.ones_like(true_doc)), ("doc", true_doc)):
                pb[key][0] += np.bincount(b[sel], weights=tok[sel], minlength=nb)
                pb[key][1] += np.bincount(b[sel], minlength=nb)
            if rec:
                for li, m in enumerate(gdn):
                    r = m.rec
                    a = r["a"].float()[0] + m.dt_bias.float()
                    # kernel: g = -exp(A_log) * softplus(a + dt_bias) (softplus = x for x >= 20), alpha = exp(g)
                    sp = torch.where(a < 20.0, torch.log1p(torch.exp(a)), a)
                    alpha = torch.exp(-torch.exp(m.A_log.float()) * sp)
                    beta = torch.sigmoid(r["b"].float()[0])
                    gate_acc["alpha"][li].append(alpha.cpu().numpy())
                    gate_acc["beta"][li].append(beta.cpu().numpy())
                    if r["s"] is not None:
                        gate_acc["score"][li].append(r["s"].float()[0].cpu().numpy())
        for m in gdn:
            m.rec = None
        loss = float(s_loss / n_tok)
        ppl[name] = {"loss": loss, "ppl": math.exp(loss), "tokens": int(n_tok)}
        pos_res[name] = {k: {"loss": (v[0] / np.maximum(v[1], 1)).tolist(), "tokens": v[1].astype(int).tolist()} for k, v in pb.items()}
        print(run.name, name, json.dumps(ppl[name]), flush=True)
    ppl["mean_loss"] = float(np.mean([ppl[s]["loss"] for s in sources]))
    json.dump({"run": run.name, "step": step, "ckpt": args.ckpt, "per_source": ppl}, open(out / "ppl.json", "w"), indent=1)
    json.dump({"run": run.name, "step": step,
               "bucket_edges": edges + [T],
               "note": "bucket j = positions [edges[j], edges[j+1]); position = index of the input token since its document start "
                       "(0 = first token, predicting the second). 'doc' only counts segments starting at a true document start; "
                       "'context' counts every doc-mask segment (row-start continuations included, position = context length).",
               "per_source": pos_res,
               "all": {k: {"loss": (sum(np.array(pos_res[s][k]["loss"]) * np.array(pos_res[s][k]["tokens"]) for s in sources)
                                    / np.maximum(sum(np.array(pos_res[s][k]["tokens"]) for s in sources), 1)).tolist(),
                           "tokens": sum(np.array(pos_res[s][k]["tokens"]) for s in sources).tolist()} for k in ("doc", "context")}},
              open(out / "position_loss.json", "w"), indent=1)

    gates = {"run": run.name, "step": step, "gate_mode": mc.gate_mode, "tokens": int(sum(a.size for a in gate_acc["beta"][0]) // mc.gdn_heads),
             "note": "alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b), as computed by fla chunk_gated_delta_rule "
                     "(use_gate_in_kernel, use_beta_sigmoid_in_kernel). log10_1m_alpha = log10(1 - alpha). "
                     "horizon = 1 / (1 - exp(-gamma)).",
             "layers": []}
    for li, m in enumerate(gdn):
        al = np.concatenate(gate_acc["alpha"][li])
        be = np.concatenate(gate_acc["beta"][li])
        sc = np.concatenate(gate_acc["score"][li]) if gate_acc["score"][li] else None
        heads = []
        for h in range(mc.gdn_heads):
            d = {"alpha": stats(al[:, h], 0.0, 1.0),
                 "log10_1m_alpha": stats(np.log10(np.maximum(1.0 - al[:, h].astype(np.float64), 1e-12)), -8.0, 0.0),
                 "beta": stats(be[:, h], 0.0, 1.0),
                 "A_log": float(m.A_log[h]), "dt_bias": float(m.dt_bias[h])}
            if m.pool:
                g = m.gamma()
                gh = 0.0 if g is None else float(g[h])
                d["gamma"] = gh
                d["horizon"] = None if gh == 0.0 else float(1.0 / -math.expm1(-gh))
                if sc is not None:
                    d["score"] = stats(sc[:, h], -mc.score_bound, mc.score_bound)
                    d["score_w_norm"] = float(m.score_w[h].float().norm())
            heads.append(d)
        lay = {"layer": li, "heads": heads}
        if m.pool:
            lay["ab_c_norm"] = float(m.ab_c.float().norm())
            lay["ab_x_norm"] = float(m.ab.weight.float().norm())
        gates["layers"].append(lay)
    json.dump(gates, open(out / "gates.json", "w"))

    tr = {"run": run.name, "params": model.num_params(), "gate_mode": mc.gate_mode, "seed": cfg["data"]["seed"]}
    logp = run / "log.jsonl"
    if logp.exists():
        recs = [json.loads(l) for l in open(logp) if l.strip()]
        steps = [r for r in recs if "tok_s" in r]
        evals = [r for r in recs if "val" in r]
        tps = [r["tok_s"] for r in steps[1:]] or [r["tok_s"] for r in steps]
        tr.update({"tok_s_mean": float(np.mean(tps)) if tps else None,
                   "tok_s_median": float(np.median(tps)) if tps else None,
                   "tok_s_note": "mean over log records excluding the first (compile/warmup); includes restarts",
                   "mem_gb_max": max((r["mem_gb"] for r in steps), default=None),
                   "final_step": steps[-1]["step"] if steps else None,
                   "final_train_loss": steps[-1]["loss"] if steps else None,
                   "elapsed_h_last_segment": steps[-1]["elapsed_h"] if steps else None,
                   "evals": evals})
    json.dump(tr, open(out / "train.json", "w"), indent=1)
    json.dump({"run": run.name, "step": step, "gate_mode": mc.gate_mode, "seed": cfg["data"]["seed"], "params": tr["params"],
               "val_mean_loss": ppl["mean_loss"], "per_source_loss": {s: ppl[s]["loss"] for s in sources},
               "tok_s_mean": tr.get("tok_s_mean")}, open(out / "summary.json", "w"), indent=1)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
