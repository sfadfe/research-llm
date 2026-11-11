"""Build a context-switch training set: rows that interleave 2-4 documents and return to earlier ones (A B C A ...).

Usage (from the research-llm directory):
    .venv/bin/python Train/build_switch_dataset.py [--train_tokens 500e6] [--val_tokens 2e6] [--seed 1234]
        -> data/switch/meta.json, data/switch/switch/{train.bin,val.bin}, data/switch/segments_{train,val}.npz

Train with the existing loader: [data] bin_dir = "data/switch", [data.mix] switch = 1.0, total_tokens <= train tokens.

Format: same as data/bin (uint16 tokens, eot = <|endoftext|>). The bin is a series of rows of seq_len tokens; row i
occupies tokens [i*T, (i+1)*T), which are exactly the chunks MixtureData reads (offset k*T, T+1 tokens). Each row is
T-1 content tokens followed by one eot, so with doc_mask the only boundary (eot position + 1) is the row end: no state
reset inside a row. One extra eot is appended at the file end so the last row is a full T+1 chunk.

Sources: train rows come from the train files of data/bin, val rows from val.bin of data/bin (disjoint documents, as
split by the tokenizer script). A "unit" is a span of <= T tokens of one document (documents longer than T are cut into
consecutive units; units < SEG_MIN tokens are dropped). Each row: k in {2,3,4} slots, k distinct sources drawn uniformly
from the 5; segment lengths uniform in [SEG_MIN, SEG_MAX] (a draw that would leave a remainder < SEG_MIN is redrawn; the
last segment is cut to fill T-1); slot labels per segment: first uniform, each next uniform among the other k-1 slots
(adjacent segments always differ), redrawn until all k slots appear; k is capped at n_segments-1 so at least one slot
returns. Each slot takes one unit (shuffled stream per source, units shorter than the slot's total length are skipped)
and its segments read the unit consecutively: a returning segment continues exactly where the slot's previous segment
stopped. Streams reshuffle when exhausted (only expected for val; counted in meta as passes).

Side file segments_<split>.npz (CSR over rows): seg_ptr (N+1), seg_start/seg_len (offset/length in the row), seg_slot
(slot in first-appearance order: 0 = A, 1 = B, ...), seg_use (index into the use table), seg_unit_off (offset of the
segment inside its unit), seg_return (slot seen earlier in the row). Use table (one entry per row slot): use_src (index
into sources), use_file (index into files[src]; -1 = val.bin), use_doc_start (file offset of the document's first
token), use_unit_start (file offset of the unit's first token), use_unit_len, use_doc_id (int64 id of the document,
unique per split: src << 56 | (file + 1) << 40 | doc_start).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["dclm", "fineweb_edu", "wikipedia", "gutenberg", "cosmopedia"]
NAME = "switch"
SEG_MIN, SEG_MAX = 64, 512
K_MIN, K_MAX = 2, 4
SCAN_CHUNK = 1 << 27
BLOCK = 4096


def log(msg):
    print(f"[build_switch] {msg}", flush=True)


def doc_id(src, file_idx, doc_start):
    """Unique id of a document within a split: source, file (val.bin -> 0), first-token offset."""
    return (np.int64(src) << 56) | (np.int64(file_idx + 1) << 40) | np.int64(doc_start)


def scan_units(path, eot, unit):
    """Units of one bin file: (doc_start, unit_start, unit_len) arrays, unit_len in [SEG_MIN, unit]."""
    m = np.memmap(path, dtype=np.uint16, mode="r")
    e = np.concatenate([np.flatnonzero(m[a:a + SCAN_CHUNK] == eot) + a for a in range(0, len(m), SCAN_CHUNK)])
    st = np.concatenate([[0], e[:-1] + 1]).astype(np.int64)
    ln = (e - st).astype(np.int64)
    nu = -(-ln // unit)
    d = np.repeat(np.arange(len(st)), nu)
    k = np.arange(int(nu.sum()), dtype=np.int64) - np.repeat(np.cumsum(nu) - nu, nu)
    us = st[d] + k * unit
    ul = np.minimum(ln[d] - k * unit, unit)
    keep = ul >= SEG_MIN
    return st[d][keep], us[keep], ul[keep]


class Pool:
    """Shuffled stream of the units of one source in one split."""

    def __init__(self, si, paths, file_ids, eot, unit, rng):
        self.si, self.rng = si, rng
        self.maps, parts = [], []
        for fi, (p, fid) in enumerate(zip(paths, file_ids)):
            ds, us, ul = scan_units(p, eot, unit)
            self.maps.append(np.memmap(p, dtype=np.uint16, mode="r"))
            parts.append((np.full(len(ul), fi, dtype=np.int32), np.full(len(ul), fid, dtype=np.int32), ds, us, ul))
        self.fidx, self.fid, self.dstart, self.ustart, self.ulen = (np.concatenate(c) for c in zip(*parts))
        self.order, self.ptr = self.rng.permutation(len(self.ulen)), 0
        self.passes, self.skipped, self.drawn = 1, 0, 0
        self.used = np.zeros(len(self.ulen), dtype=np.int32)
        self.max_len = int(self.ulen.max())

    def draw(self, need):
        assert need <= self.max_len, (SOURCES[self.si], need, self.max_len)
        while True:
            if self.ptr == len(self.order):
                self.order, self.ptr = self.rng.permutation(len(self.ulen)), 0
                self.passes += 1
            i = int(self.order[self.ptr])
            self.ptr += 1
            if self.ulen[i] >= need:
                self.used[i] += 1
                self.drawn += 1
                return i
            self.skipped += 1

    def read(self, i, off, n):
        return self.maps[self.fidx[i]][self.ustart[i] + off:self.ustart[i] + off + n]

    def stats(self):
        return {"units": int(len(self.ulen)), "unit_tokens": int(self.ulen.sum()), "drawn": self.drawn,
                "skipped_too_short": self.skipped, "passes": self.passes,
                "units_used_more_than_once": int((self.used > 1).sum()), "max_unit_use": int(self.used.max())}


def seg_lengths(rng, total):
    lens, cum = [], 0
    while cum < total:
        n = int(rng.integers(SEG_MIN, SEG_MAX + 1))
        if cum + n >= total:
            n = total - cum
        elif total - cum - n < SEG_MIN:
            continue
        lens.append(n)
        cum += n
    return lens


def slot_labels(rng, n, k):
    while True:
        lab = [int(rng.integers(k))]
        for _ in range(n - 1):
            lab.append((lab[-1] + int(rng.integers(1, k))) % k)
        if len(set(lab)) == k:
            first = {}
            for s in lab:
                first.setdefault(s, len(first))
            return [first[s] for s in lab]


def build_split(split, pools, n_rows, T, eot, rng, out_bin):
    content = T - 1
    seg = {k: [] for k in ("start", "len", "slot", "use", "unit_off", "ret")}
    seg_ptr = [0]
    use = {k: [] for k in ("src", "file", "doc_start", "unit_start", "unit_len")}
    k_hist = np.zeros(K_MAX + 1, dtype=np.int64)
    tmp = Path(str(out_bin) + ".tmp")
    t0 = time.time()
    with open(tmp, "wb") as f:
        buf = np.empty((BLOCK, T), dtype=np.uint16)
        nb = 0
        for r in range(n_rows):
            lens = seg_lengths(rng, content)
            k = min(int(rng.integers(K_MIN, K_MAX + 1)), len(lens) - 1)
            lab = slot_labels(rng, len(lens), k)
            srcs = rng.choice(len(SOURCES), size=k, replace=False)
            need = np.bincount(lab, weights=lens, minlength=k).astype(np.int64)
            u0 = len(use["src"])
            units = []
            for j in range(k):
                p = pools[int(srcs[j])]
                i = p.draw(int(need[j]))
                units.append(i)
                use["src"].append(int(srcs[j]))
                use["file"].append(int(p.fid[i]))
                use["doc_start"].append(int(p.dstart[i]))
                use["unit_start"].append(int(p.ustart[i]))
                use["unit_len"].append(int(p.ulen[i]))
            cur = [0] * k
            pos = 0
            row = buf[nb]
            for n, j in zip(lens, lab):
                row[pos:pos + n] = pools[int(srcs[j])].read(units[j], cur[j], n)
                seg["start"].append(pos)
                seg["len"].append(n)
                seg["slot"].append(j)
                seg["use"].append(u0 + j)
                seg["unit_off"].append(cur[j])
                seg["ret"].append(cur[j] > 0)
                cur[j] += n
                pos += n
            assert pos == content
            row[content] = eot
            seg_ptr.append(len(seg["start"]))
            k_hist[k] += 1
            nb += 1
            if nb == BLOCK or r == n_rows - 1:
                buf[:nb].tofile(f)
                nb = 0
            if (r + 1) % 20000 == 0 or r == n_rows - 1:
                log(f"{split}: {r + 1}/{n_rows} rows {time.time() - t0:.0f}s")
        np.array([eot], dtype=np.uint16).tofile(f)
    os.replace(tmp, out_bin)
    side = {"seg_ptr": np.asarray(seg_ptr, dtype=np.int64),
            "seg_start": np.asarray(seg["start"], dtype=np.int32), "seg_len": np.asarray(seg["len"], dtype=np.int32),
            "seg_slot": np.asarray(seg["slot"], dtype=np.int8), "seg_use": np.asarray(seg["use"], dtype=np.int64),
            "seg_unit_off": np.asarray(seg["unit_off"], dtype=np.int32),
            "seg_return": np.asarray(seg["ret"], dtype=bool),
            "use_src": np.asarray(use["src"], dtype=np.int8), "use_file": np.asarray(use["file"], dtype=np.int32),
            "use_doc_start": np.asarray(use["doc_start"], dtype=np.int64),
            "use_unit_start": np.asarray(use["unit_start"], dtype=np.int64),
            "use_unit_len": np.asarray(use["unit_len"], dtype=np.int32)}
    side["use_doc_id"] = np.array([doc_id(s, f, d) for s, f, d in
                                   zip(side["use_src"], side["use_file"], side["use_doc_start"])], dtype=np.int64)
    return side, k_hist


def split_stats(side, k_hist, n_rows, T):
    nseg = np.diff(side["seg_ptr"])
    ret = side["seg_return"]
    content = n_rows * (T - 1)
    ret_tok = int(side["seg_len"][ret].sum())
    rows_ret = np.add.reduceat(ret.astype(np.int64), side["seg_ptr"][:-1])
    return {"rows": n_rows, "tokens": n_rows * T, "content_tokens": content,
            "segments_per_row": {"mean": float(nseg.mean()), "min": int(nseg.min()), "max": int(nseg.max()),
                                 "hist": {int(v): int(c) for v, c in zip(*np.unique(nseg, return_counts=True))}},
            "docs_per_row_hist": {k: int(k_hist[k]) for k in range(K_MIN, K_MAX + 1)},
            "returning_segments": int(ret.sum()), "returning_segments_per_row_mean": float(rows_ret.mean()),
            "rows_without_return": int((rows_ret == 0).sum()),
            "share_tokens_in_returning_segments": ret_tok / content,
            "segment_len_mean": float(side["seg_len"].mean()),
            "returning_segment_len_mean": float(side["seg_len"][ret].mean()),
            "unique_docs": int(len(np.unique(side["use_doc_id"]))), "uses": int(len(side["use_doc_id"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "data" / "bin"))
    ap.add_argument("--out", default=str(ROOT / "data" / "switch"))
    ap.add_argument("--train_tokens", type=float, default=500e6)
    ap.add_argument("--val_tokens", type=float, default=2e6)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    T = args.seq_len
    src = Path(args.src)
    meta_in = json.load(open(src / "meta.json"))
    eot = int(meta_in["eot"])
    out = Path(args.out)
    (out / NAME).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = {"tokenizer": meta_in["tokenizer"], "dtype": "uint16", "eot": eot, "vocab_size": meta_in["vocab_size"],
            "sources": {NAME: {"train": [], "val": None}}}
    info = {"seed": args.seed, "seq_len": T, "src_bin_dir": str(src), "sources": SOURCES,
            "seg_len_range": [SEG_MIN, SEG_MAX], "docs_per_row_range": [K_MIN, K_MAX], "unit_max_len": T,
            "files": {s: [fn for fn, _ in meta_in["sources"][s]["train"]] for s in SOURCES},
            "row_layout": "T-1 content tokens + eot; file ends with one extra eot", "splits": {}}
    for si_split, split in enumerate(("val", "train")):
        n_rows = -(-int(getattr(args, f"{split}_tokens")) // T)
        rng = np.random.default_rng([args.seed, si_split])
        pools = []
        for si, s in enumerate(SOURCES):
            if split == "train":
                files = [fn for fn, _ in meta_in["sources"][s]["train"]]
                paths, fids = [src / s / fn for fn in files], list(range(len(files)))
            else:
                paths, fids = [src / s / "val.bin"], [-1]
            pools.append(Pool(si, paths, fids, eot, T, np.random.default_rng([args.seed, si_split, si])))
            log(f"{split} {s}: {len(pools[-1].ulen)} units, {pools[-1].ulen.sum() / 1e6:.1f}M tokens "
                f"{time.time() - t0:.0f}s")
        out_bin = out / NAME / f"{split}.bin"
        side, k_hist = build_split(split, pools, n_rows, T, eot, rng, out_bin)
        np.savez(out / f"segments_{split}.npz", **side)
        st = split_stats(side, k_hist, n_rows, T)
        st["pools"] = {s: p.stats() for s, p in zip(SOURCES, pools)}
        st["file_tokens"] = n_rows * T + 1
        info["splits"][split] = st
        if split == "train":
            meta["sources"][NAME]["train"] = [["train.bin", n_rows * T + 1]]
            meta["sources"][NAME]["train_tokens"] = n_rows * T + 1
        else:
            meta["sources"][NAME]["val"] = ["val.bin", n_rows * T + 1]
        log(f"{split} done: {json.dumps({k: v for k, v in st.items() if k != 'pools'})}")
        log(f"{split} pools: {json.dumps(st['pools'])}")
    meta[NAME] = info
    with open(out / "meta.json.tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(out / "meta.json.tmp", out / "meta.json")
    log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
