import json
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

from model import LM, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
T = 2048
MB = 8


def tensors(buf, eot, device):
    x, y = buf[:, :-1], buf[:, 1:]
    B, L = x.shape
    pos = np.flatnonzero(x.reshape(-1) == eot) + 1
    cu_cpu = torch.from_numpy(np.union1d(np.arange(0, B * L + 1, L), pos))
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return to(x), to(y), cu_cpu.to(device), cu_cpu


@torch.no_grad()
def nll(model, batches, eot, device):
    s, n = 0.0, 0
    for buf, y_mask in batches:
        x, y, cu, cu_cpu = tensors(buf, eot, device)
        if y_mask is not None:
            y = torch.where(torch.from_numpy(np.ascontiguousarray(y_mask)).to(device), y, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, cu, cu_cpu).float().item()
        w = int((y != -100).sum())
        s += loss * w
        n += w
    return s / n, n


def pretrain_batches(name):
    m = np.memmap(ROOT / "data/bin" / name / "val.bin", dtype=np.uint16, mode="r")
    k = (len(m) - 1) // T
    k -= k % MB
    for i in range(0, k, MB):
        yield np.stack([m[o * T:o * T + T + 1] for o in range(i, i + MB)]).astype(np.int64), None


def sft_batches():
    meta = json.load(open(ROOT / "data/sft/meta.json"))
    R = meta["row"]
    tok = np.memmap(ROOT / "data/sft/val.bin", dtype=np.uint16, mode="r").reshape(-1, R)
    flg = np.memmap(ROOT / "data/sft/val_mask.bin", dtype=np.uint8, mode="r").reshape(-1, R)
    for i in range(0, len(tok), MB):
        yield tok[i:i + MB].astype(np.int64), flg[i:i + MB, 1:] > 0


def main():
    device = torch.device("cuda", 0)
    sources = list(json.load(open(ROOT / "data/bin/meta.json"))["sources"])
    eot = json.load(open(ROOT / "data/sft/meta.json"))["eot"]
    res = {}
    for run in sys.argv[1:]:
        run = Path(run)
        cfg = tomllib.load(open(run / "config.toml", "rb"))
        model = LM(ModelConfig(**cfg["model"])).to(device).eval()
        model.load_state_dict(torch.load(run / "latest.pt", map_location=device, weights_only=False)["model"])
        r = {}
        for s in sources:
            l, n = nll(model, pretrain_batches(s), eot, device)
            r[s] = {"loss": l, "ppl": math.exp(l), "tokens": n}
            print(run.name, s, json.dumps(r[s]), flush=True)
        l, n = nll(model, sft_batches(), eot, device)
        r["sft_val"] = {"loss": l, "ppl": math.exp(l), "tokens": n}
        print(run.name, "sft_val", json.dumps(r["sft_val"]), flush=True)
        res[run.name] = r
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(ROOT / "model" / "ppl.json", "w"), indent=1)


if __name__ == "__main__":
    main()
