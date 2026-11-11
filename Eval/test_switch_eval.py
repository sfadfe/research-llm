"""Checks for Eval/switch_eval.py on a real checkpoint (small, < 1 min).

Usage (from research-llm): .venv/bin/python Eval/test_switch_eval.py --run model/<run dir> --device cuda:N [--results DIR]
 1. rows: deterministic for a seed; Exp 1 segments are the first 512 tokens of docs, consecutive sources differ;
    Exp 2 A >= 1024 tokens, B from another source; bucket token counts [1,1,2,...,128,255].
 2. statistics: cluster_se with one observation per cluster = sample std / sqrt(n); retained() on constructed data.
 3. per-token loss of packed rows matches LM.forward's fused loss.
 4. resets: segment j of a reset row (a) = the same segment run alone (up to unbiased bf16 packing noise); A1|A2 with a reset at 512 = A2 alone (iii);
    segment 0 of (b) = segment 0 of (a); no reset changes segments 1-3 (state does carry).
 5. DIR (results/switch/<run> written by Eval/switch_eval.py): every JSON parses with NaN/Infinity rejected and the
    recorded segment-0 check is ok.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Eval"))
import switch_eval as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
    ap.add_argument("--results", default=None, help="results/switch/<run> dir written by Eval/switch_eval.py")
    args = ap.parse_args()
    eot = int(json.load(open(S.ROOT / "data/bin/meta.json"))["eot"])
    docs = S.load_docs(eot)

    # 1. rows
    n = 40
    src, doc = S.build_exp1(docs, n, 0)
    src2, doc2 = S.build_exp1(docs, n, 0)
    e2 = S.build_exp2(docs, n, 0)
    check("rows_deterministic", np.array_equal(doc, doc2) and np.array_equal(src, src2)
          and all(np.array_equal(a, b) for a, b in zip(e2, S.build_exp2(docs, n, 0))), "same seed -> same rows")
    ok = all((src[:, j] != src[:, j + 1]).all() for j in range(S.NSEG - 1))
    ok &= all(docs[S.SOURCES[s]][2][d] >= S.SEG for s, d in zip(src.ravel(), doc.ravel()))
    m, st, _ = docs[S.SOURCES[src[3, 2]]]
    d = doc[3, 2]
    ok &= (st[d] == 0 or m[st[d] - 1] == eot) and np.array_equal(S.toks(docs, src[3, 2], d, 0, S.SEG), m[st[d]:st[d] + S.SEG])
    check("exp1_rows", bool(ok), "segments start at doc starts, >= 512 tokens, consecutive sources differ")
    a_src, a_doc, b_src, b_doc = e2
    ok = (a_src != b_src).all() and all(docs[S.SOURCES[s]][2][d] >= 2 * S.SEG for s, d in zip(a_src, a_doc))
    ok &= all(docs[S.SOURCES[s]][2][d] >= max(S.LS) for s, d in zip(b_src, b_doc))
    check("exp2_rows", bool(ok), "A >= 1024 tokens, B >= max(L) tokens, B source != A source")
    check("bucket_counts", S.CNT.astype(int).tolist() == [1, 1, 2, 4, 8, 16, 32, 64, 128, 255], str(S.CNT.astype(int).tolist()))

    # 2. statistics
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    se = S.cluster_se(x, np.arange(50))
    check("cluster_se_iid", np.allclose(se, x.std(0, ddof=1) / np.sqrt(50)), f"{se} vs std/sqrt(n)")
    i = np.full((10, S.NB), 2.0)
    iii = np.full((10, S.NB), 4.0)
    ii = np.full((10, S.NB), 2.5)
    r = S.retained(i, ii, iii)
    check("retained_value", np.allclose(r, 0.75), f"(4-2.5)/(4-2) = {r[0]}")

    # 3-4. model
    run = Path(args.run).resolve()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    model, _, ckpt, _ = S.load_model(run, dev)
    print(f"run {run.name} ckpt {ckpt.name} mode {model.c.gate_mode}", flush=True)
    tl = S.TokenLoss(model, dev)
    k = 4
    rows = [np.concatenate([S.toks(docs, src[r, j], doc[r, j], 0, S.SEG) for j in range(S.NSEG)]) for r in range(k)]
    cuts = [j * S.SEG for j in range(1, S.NSEG)]
    ta = tl(rows, [cuts] * k)
    tb = tl(rows, [[]] * k)
    check("tokenwise_vs_fused", tl.max_dev < 5e-3, f"max |mean per-token CE - fused loss| {tl.max_dev:.2e}")
    alone = tl([rows[r][j * S.SEG:(j + 1) * S.SEG] for r in range(k) for j in range(S.NSEG)], [[]] * (k * S.NSEG))
    # packing changes GEMM shapes, so bf16 rounding differs per token (mean |diff| ~1e-2, measured on both runs; the
    # same packing twice is bit-identical): check the bias (mean signed diff) and the size of the noise
    d = np.stack([ta[r][j * S.SEG:j * S.SEG + S.SEG - 1] - alone[r * S.NSEG + j][:S.SEG - 1]
                  for r in range(k) for j in range(S.NSEG)])
    check("reset_equals_alone", abs(d.mean()) < 3e-3 and np.abs(d).mean() < 2e-2,
          f"per-token CE (a) - segment alone: mean {d.mean():+.2e}, mean abs {np.abs(d).mean():.2e} (packing noise)")
    d0 = max(float(np.abs(ta[r][:S.SEG - 1] - tb[r][:S.SEG - 1]).max()) for r in range(k))
    check("segment0_b_equals_a", d0 < 2e-2, f"max per-token |(b)-(a)| on segment 0 {d0:.2e}")
    d1 = min(float(np.abs(ta[r][S.SEG:] - tb[r][S.SEG:]).mean()) for r in range(k))
    check("no_reset_carries", d1 > 1e-2, f"min over rows of mean |(b)-(a)| on segments 1-3 {d1:.2e}")
    a1a2 = [np.concatenate([S.toks(docs, a_src[r], a_doc[r], 0, 2 * S.SEG)]) for r in range(k)]
    tr = tl(a1a2, [[S.SEG]] * k)
    t3 = tl([x[S.SEG:] for x in a1a2], [[]] * k)
    d = np.stack([tr[r][S.SEG:S.SEG + S.SEG - 1] - t3[r][:S.SEG - 1] for r in range(k)])
    check("exp2_reset_equals_iii", abs(d.mean()) < 3e-3 and np.abs(d).mean() < 2e-2,
          f"A1|A2 with reset at 512 - A2 alone: mean {d.mean():+.2e}, mean abs {np.abs(d).mean():.2e} (packing noise)")
    tl.close()

    # 5. results
    if args.results:
        res = Path(args.results)

        def bad_const(c):
            raise ValueError(f"non-finite constant {c}")
        for f in sorted(res.glob("*.json")):
            try:
                json.load(open(f), parse_constant=bad_const)
                ok, msg = True, "valid, finite"
            except Exception as e:  # noqa: BLE001
                ok, msg = False, str(e)
            check(f"json:{f.name}", ok, msg)
        e1 = json.load(open(res / "exp1_switch.json"))
        check("results_segment0", e1["check_segment0"]["ok"], json.dumps(e1["check_segment0"]))
    print("max mem GB", torch.cuda.max_memory_allocated(dev) / 2 ** 30)
    print("FAILED: " + ", ".join(fails) if fails else "ALL PASS", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
