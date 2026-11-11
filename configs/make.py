"""Generate the run configs. Usage: python configs/make.py [--smoke STEPS] [--micro N] [--outdir DIR]"""
import argparse
from pathlib import Path

ARMS = {"A": "linear", "B": "pool", "C": "pool_g0", "D": "pool_s0"}
RUNS = [("A", 1234), ("B", 1234), ("C", 1234), ("D", 1234), ("A", 4321), ("B", 4321)]

TEMPLATE = """[model]
vocab_size = 32768
d_model = 768
n_layers = 12
pattern = "G"
mlp_hidden = 2048
norm_eps = 1e-6
gdn_heads = 6
gdn_head_dim = 128
conv_size = 4
init_std = 0.02
loss = "fused"
gate_mode = "{mode}"
pool_chunk = 64
pool_ckpt = true
pool_h_min = 4.0
pool_h_max = 256.0
score_bound = 10.0

[data]
bin_dir = "data/bin"
seq_len = 2048
total_tokens = 2_000_000_000
doc_mask = true
seed = {seed}

[data.mix]
dclm = 0.45
fineweb_edu = 0.25
wikipedia = 0.10
gutenberg = 0.10
cosmopedia = 0.10

[train]
run_name = "{name}"
micro_batch = {micro}
global_batch = 128
lr = 1e-3
min_lr_ratio = 0.0
warmup_steps = 500
decay_frac = 0.2
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
compile = true
max_steps = {max_steps}
log_every = {log_every}
eval_every = {eval_every}
eval_batches = {eval_batches}
save_every = {save_every}
keep_every = 0
"""


def name(arm, seed):
    return f"{arm}_{ARMS[arm]}_s{seed}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [(arm, 1234) for arm in ARMS] if a.smoke else RUNS
    for arm, seed in runs:
        n = name(arm, seed) + ("_smoke" if a.smoke else "")
        kw = dict(mode=ARMS[arm], seed=seed, name=n, micro=a.micro, max_steps=a.smoke, log_every=5,
                  eval_every=500, eval_batches=16, save_every=250)
        if a.smoke:
            kw.update(log_every=1, eval_every=a.smoke, eval_batches=2, save_every=a.smoke)
        (out / f"{n}.toml").write_text(TEMPLATE.format(**kw))
        print(out / f"{n}.toml")
