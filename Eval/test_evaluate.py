"""Checks for Eval/evaluate.py on a real checkpoint (small, ~2 GB GPU, < 1 min).

Usage (from research-llm): .venv/bin/python Eval/test_evaluate.py --run model/<run dir> --device cuda:N [--results DIR]
 1. alpha/beta: per layer, the kernel re-run with use_gate_in_kernel=False / use_beta_sigmoid_in_kernel=False and
    g = log_alpha(a), beta = beta_of(b) from Eval/evaluate.py reproduces the model's kernel output; fla's naive
    recurrent reference with those alpha/beta reproduces it too (first 256 tokens of the first segment); substituting
    all layers end to end reproduces the loss.
 2. pooled arms: window_weights(s, gamma, seg) applied to x = A_h [k; v] (c_proj) reproduces last_gate['c'] (pooled context).
 3. loss: Train/train.py evaluate() on the same rows (first 2 * micro_batch rows per source) vs Eval/evaluate.py
    ppl.json in DIR (run with --max-rows 2*micro_batch), and vs log.jsonl's val at the checkpoint step.
 4. every JSON in DIR parses with NaN/Infinity rejected.
"""
import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Eval"))
import evaluate as E  # noqa: E402  (sets HIPBLASLT path, imports Train/model.py as E.M)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule  # noqa: E402

M = E.M
fails = []


def check(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
    if not ok:
        fails.append(name)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default=None, help="results/<run> dir written by Eval/evaluate.py --max-rows 2*micro")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    cfg = tomllib.load(open(run / "config.toml", "rb"))
    T = cfg["data"]["seq_len"]
    eot = json.load(open(ROOT / "data/bin/meta.json"))["eot"]
    ckpt = E.pick_ckpt(run, None)
    model = M.LM(M.ModelConfig(**cfg["model"])).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    step = ck["step"]
    del ck
    gdn = [b.mixer for b in model.blocks if b.kind == "G"]
    print(f"run {run.name} ckpt {ckpt.name} mode {model.c.gate_mode}", flush=True)

    # one real batch of 2 rows (dclm), doc-masked as in training
    mm, _ = E.val_rows("dclm", T)
    buf = np.stack([mm[o * T:o * T + T + 1] for o in range(2)]).astype(np.int64)
    x, y, cu, cu_cpu = E.tensors(buf, eot, dev)

    calls = []
    orig = M.chunk_gated_delta_rule

    def tap(*a, **kw):
        o, st = orig(*a, **kw)
        calls.append((a, kw, o))
        return o, st

    M.chunk_gated_delta_rule = tap
    for m in gdn:
        m.record_gate = True
    rec = []
    hs = [m.register_forward_hook(lambda mod, i, o: rec.append(mod.last_gate)) for m in gdn]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss0 = float(model(x, y, cu, cu_cpu))
    for h in hs:
        h.remove()
    for m in gdn:
        m.record_gate = False
    M.chunk_gated_delta_rule = orig

    # 1a. per layer: kernel with precomputed alpha/beta vs kernel's own gate path
    worst = worst_n = worst_mean = worst_frac = 0.0
    for li, ((q, k, v), kw, o) in enumerate(calls):
        m = gdn[li]
        assert torch.equal(kw["g"], rec[li]["a"]) and torch.equal(kw["beta"], rec[li]["b"])
        la = E.log_alpha(kw["g"], kw["A_log"], kw["dt_bias"])
        be = E.beta_of(kw["beta"])
        kw2 = dict(kw, g=la, beta=be, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, A_log=None, dt_bias=None)
        o2, _ = orig(q, k, v, **kw2)
        d = (o2.float() - o.float()).abs()
        worst = max(worst, float(d.max() / o.float().abs().max()))
        worst_mean = max(worst_mean, float(d.mean() / o.float().abs().mean()))
        worst_frac = max(worst_frac, float((d > 0).float().mean()))
        # 1b. naive fp32 recurrence (independent of the chunk kernel), first 256 tokens of the first segment
        n = min(256, int(cu_cpu[1]))
        qn = F.normalize(q[:, :n].float(), dim=-1)
        kn = F.normalize(k[:, :n].float(), dim=-1)
        on, _ = naive_recurrent_gated_delta_rule(qn, kn, v[:, :n].float(), be[:, :n], la[:, :n])
        errn = float((on - o[:, :n].float()).abs().max() / o[:, :n].float().abs().max())
        worst_n = max(worst_n, errn)
    # bf16 output: an exact formula leaves only rounding flips (a 1% change of log alpha or beta gives mean rel ~6e-3,
    # >70% of elements differing; dropping dt_bias gives ~1)
    check("alpha_beta_kernel", worst_mean < 1e-3 and worst_frac < 0.2,
          f"kernel own gates vs precomputed alpha/beta: mean rel err {worst_mean:.2e}, max rel err {worst:.2e}, "
          f"fraction of bf16 outputs differing {worst_frac:.3f}")
    check("alpha_beta_naive", worst_n < 3e-2, f"max rel err (fla naive recurrence with our alpha/beta, bf16 inputs) {worst_n:.2e}")

    # 1c. end to end: all layers fed precomputed alpha/beta
    def sub(q, k, v, **kw):
        la = E.log_alpha(kw["g"], kw["A_log"], kw["dt_bias"])
        kw = dict(kw, g=la, beta=E.beta_of(kw["beta"]), use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                  A_log=None, dt_bias=None)
        return orig(q, k, v, **kw)

    M.chunk_gated_delta_rule = sub
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss1 = float(model(x, y, cu, cu_cpu))
    M.chunk_gated_delta_rule = orig
    check("alpha_beta_end_to_end", abs(loss1 - loss0) < 2e-3, f"loss kernel gates {loss0:.6f} vs precomputed {loss1:.6f}")

    # 2. pooled arms: realised window weights reproduce the pooled context c
    if gdn[0].pool:
        worst = 0.0
        seg = torch.zeros(x.numel(), dtype=torch.long, device=dev)
        seg[cu[1:-1].long()] = 1
        seg = seg.cumsum(0)
        for li, ((q, k, v), kw, o) in enumerate(calls):
            r = rec[li]
            H = gdn[li].h
            s = r["s"][0].float() if r["s"] is not None else torch.zeros(x.numel(), H, device=dev)
            z = torch.cat([k, v], -1)[0].float()
            z = torch.einsum("thd,hrd->thr", z, gdn[li].c_proj.float().view(H, gdn[li].rank, -1))
            for row in range(2):
                sl = slice(row * T, (row + 1) * T)
                p, _ = E.window_weights(s[sl], r["gamma"].float(), seg[sl], H)
                c = torch.einsum("hti,ihd->thd", p, z[sl])
                ref = r["c"][0, sl].float()
                worst = max(worst, float((c - ref).abs().max() / ref.abs().max()))
        check("pool_window_weights", worst < 1e-3, f"max rel err of sum_i w_ti x_i vs model c_t {worst:.2e}")

    # 3. loss vs Train/train.py evaluate on the same rows
    sys.path.insert(0, str(ROOT / "Train"))
    from data import MixtureData
    from train import evaluate as train_evaluate
    mb = cfg["train"]["micro_batch"]
    nbat = 2
    data = MixtureData(ROOT / cfg["data"]["bin_dir"], cfg["data"]["mix"], T, cfg["data"]["total_tokens"],
                       cfg["data"]["seed"], index_path=run / "index.npy", load=True)
    small = 8 if mb > 8 else mb  # same rows (0 .. 2*mb-1) in smaller batches to keep memory low
    tr = train_evaluate(model, data, small, nbat * mb // small, 0, 1, dev, True)
    print("train.py evaluate:", json.dumps(tr), flush=True)
    logv = [json.loads(l) for l in open(run / "log.jsonl") if '"val"' in l]
    logv = [r for r in logv if r["step"] == step]
    if logv and cfg["train"]["eval_batches"] == nbat:
        d = max(abs(logv[-1]["val"][s] - tr[s]) for s in tr)
        check("train_evaluate_vs_log", d < 5e-3, f"max |train.py evaluate now - log.jsonl val at step {step}| = {d:.2e}")
    if args.results:
        res = Path(args.results)
        pj = json.load(open(res / "ppl.json"))
        if all(pj["per_source"][s]["rows"] == nbat * mb for s in tr):
            d = max(abs(pj["per_source"][s]["loss"] - tr[s]) for s in tr)
            check("eval_vs_train_evaluate", d < 5e-3, f"max |Eval ppl.json loss - train.py evaluate| = {d:.2e} "
                  f"over {nbat * mb} rows/source")
        else:
            check("eval_vs_train_evaluate", False, f"ppl.json rows {[pj['per_source'][s]['rows'] for s in tr]} != {nbat * mb}")
        # 4. JSON validity / finiteness
        def bad_const(c):
            raise ValueError(f"non-finite constant {c}")
        for f in sorted(res.glob("*.json")):
            try:
                json.load(open(f), parse_constant=bad_const)
                ok, msg = True, "valid, finite"
            except Exception as e:  # noqa: BLE001
                ok, msg = False, str(e)
            check(f"json:{f.name}", ok, msg)
        sm = json.load(open(res / "summary.json"))
        check("summary_fields", all(sm.get(k) is not None for k in ("arm", "seed", "params", "val_loss", "val_mean_loss",
                                                                     "tok_s_mean")), str({k: sm.get(k) for k in ("arm", "seed", "params", "val_mean_loss", "tok_s_mean")}))
    print("max mem GB", torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
