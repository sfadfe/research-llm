import json
from pathlib import Path

import numpy as np
import torch


class MixtureData:
    def __init__(self, bin_dir, mix, seq_len, total_tokens, seed, index_path=None, load=False, write=False):
        self.dir = Path(bin_dir)
        with open(self.dir / "meta.json") as f:
            self.meta = json.load(f)
        self.eot = self.meta["eot"]
        self.T = seq_len
        self.names = list(mix)
        self.paths = [self.dir / n / fname for n in self.names for fname, _ in self.meta["sources"][n]["train"]]
        self.maps = {}
        self.stats = {}
        if load:
            self.index = np.load(index_path)
        else:
            self.index = self.build(mix, total_tokens, seed)
            if write:
                np.save(str(index_path) + ".tmp.npy", self.index)
                Path(str(index_path) + ".tmp.npy").replace(index_path)

    def build(self, mix, total_tokens, seed):
        rng = np.random.default_rng(seed)
        rows, fi = [], 0
        for name in self.names:
            avail = []
            for _, n in self.meta["sources"][name]["train"]:
                m = max((n - 1) // self.T, 0)
                avail.append(np.stack([np.full(m, fi, dtype=np.int64), np.arange(m, dtype=np.int64) * self.T], 1))
                fi += 1
            avail = np.concatenate(avail)
            need = int(round(total_tokens * mix[name] / self.T))
            reps = -(-need // len(avail))
            pick = np.concatenate([rng.permutation(len(avail)) for _ in range(reps)])[:need]
            rows.append(avail[pick])
            self.stats[name] = {"chunks": need, "available": len(avail), "epochs": need / len(avail)}
        index = np.concatenate(rows)
        return index[rng.permutation(len(index))]

    def num_steps(self, global_batch):
        return len(self.index) // global_batch

    def _map(self, fi):
        m = self.maps.get(fi)
        if m is None:
            m = self.maps[fi] = np.memmap(self.paths[fi], dtype=np.uint16, mode="r")
        return m

    def gather(self, rows):
        buf = np.empty((len(rows), self.T + 1), dtype=np.int64)
        for i, (fi, off) in enumerate(rows):
            buf[i] = self._map(fi)[off:off + self.T + 1]
        return buf

    def train_buf(self, step, micro, rank, world, mb, accum):
        start = step * mb * accum * world + (micro * world + rank) * mb
        return self.gather(self.index[start:start + mb])

    def val_bufs(self, name, mb, n_batches, rank, world):
        m = np.memmap(self.dir / name / "val.bin", dtype=np.uint16, mode="r")
        k = min((len(m) - 1) // self.T, n_batches * mb * world)
        starts = np.arange(k, dtype=np.int64)[rank::world] * self.T
        out = []
        for i in range(len(starts) // mb):
            s = starts[i * mb:(i + 1) * mb]
            out.append(np.stack([m[o:o + self.T + 1] for o in s]).astype(np.int64))
        return out

    def tensors(self, buf, device, doc_mask):
        x = torch.from_numpy(np.ascontiguousarray(buf[:, :-1])).pin_memory().to(device, non_blocking=True)
        y = torch.from_numpy(np.ascontiguousarray(buf[:, 1:])).pin_memory().to(device, non_blocking=True)
        if not doc_mask:
            return x, y, None, None
        B, T = buf.shape[0], buf.shape[1] - 1
        pos = np.flatnonzero(buf[:, :-1].reshape(-1) == self.eot) + 1
        cu_cpu = torch.from_numpy(np.union1d(np.arange(0, B * T + 1, T), pos))
        return x, y, cu_cpu.to(device), cu_cpu
