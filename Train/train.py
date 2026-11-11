import argparse
import faulthandler
import json
import os
import shutil
import signal
import time
import tomllib
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import MixtureData
from model import LM, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def lr_at(step, total, tc):
    peak = tc["lr"]
    floor = peak * tc["min_lr_ratio"]
    warm = tc["warmup_steps"]
    decay = max(int(total * tc["decay_frac"]), 1)
    if step < warm:
        return peak * (step + 1) / warm
    if step < total - decay:
        return peak
    return floor + (peak - floor) * (total - step) / decay


def save(run, model, opt, step, keep):
    ck = {"model": model.state_dict(), "optim": opt.state_dict(), "step": step}
    tmp = run / "latest.pt.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, run / "latest.pt")
    if keep:
        torch.save({"model": ck["model"], "step": step}, run / f"step_{step:06d}.pt")


@torch.no_grad()
def evaluate(model, data, mb, n_batches, rank, world, device, doc_mask):
    model.eval()
    out = {}
    for name in data.names:
        acc = torch.zeros(2, device=device)
        for buf in data.val_bufs(name, mb, n_batches, rank, world):
            x, y, cu, cu_cpu = data.tensors(buf, device, doc_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y, cu, cu_cpu)
            acc += torch.stack([loss.float(), torch.ones((), device=device)])
        if world > 1:
            dist.all_reduce(acc)
        out[name] = (acc[0] / acc[1].clamp(min=1)).item()
    model.train()
    return out


def apply_grads(model, opt, grad_clip):
    gnorm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
    if torch.isfinite(gnorm):
        torch.nn.utils.clip_grads_with_norm_(model.parameters(), grad_clip, gnorm)
        opt.step()
        return gnorm, None
    top = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            top.append(((~p.grad.isfinite()).sum().item(), p.grad.nan_to_num(0.0, 0.0, 0.0).abs().max().item(), n))
    return gnorm, sorted(top, reverse=True)[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Train" / "config.toml"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=str(ROOT / "model"))
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if ddp:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    master = rank == 0

    if args.resume:
        run = Path(args.resume)
        cfg = load_config(run / "config.toml")
    else:
        cfg = load_config(args.config)
        name = [f"{cfg['train']['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"]
        if ddp:
            dist.broadcast_object_list(name, src=0)
        run = Path(args.out) / name[0]
        if master:
            run.mkdir(parents=True, exist_ok=True)
            shutil.copy(args.config, run / "config.toml")
        if ddp:
            dist.barrier()
    tc, dc = cfg["train"], cfg["data"]

    data = MixtureData(ROOT / dc["bin_dir"], dc["mix"], dc["seq_len"], dc["total_tokens"], dc["seed"],
                       index_path=run / "index.npy", load=bool(args.resume), write=master and not args.resume)
    mb = tc["micro_batch"]
    accum = tc["global_batch"] // (mb * world)
    assert accum * mb * world == tc["global_batch"]
    total = data.num_steps(tc["global_batch"])
    if tc.get("max_steps"):
        total = min(total, tc["max_steps"])

    torch.manual_seed(dc["seed"])
    mc = ModelConfig(**cfg["model"])
    model = LM(mc).to(device)
    opt = torch.optim.AdamW(model.param_groups(tc["weight_decay"]), lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
                            eps=1e-8, fused=True)
    step = 0
    if args.resume:
        ck = torch.load(run / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        step = ck["step"]
        del ck
    if tc["compile"]:
        model.compile_mlps()
    net = DDP(model, device_ids=[local]) if ddp else model

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    n_params = model.num_params()
    tok_step = tc["global_batch"] * dc["seq_len"]
    log = open(run / "log.jsonl", "a") if master else None
    if master:
        print(f"run {run}", flush=True)
        print(f"params {n_params / 1e6:.1f}M  layers {''.join(mc.kinds())}  world {world}  micro {mb}  accum {accum}  "
              f"steps {step}->{total}  tokens/step {tok_step}  total {total * tok_step / 1e9:.2f}B", flush=True)
        if data.stats:
            print(json.dumps(data.stats), flush=True)

    loss_acc = torch.zeros((), device=device)
    n_acc = 0
    t_log, step_log, t_start = time.time(), step, time.time()
    while step < total:
        lr = lr_at(step, total, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        for i in range(accum):
            buf = data.train_buf(step, i, rank, world, mb, accum)
            x, y, cu, cu_cpu = data.tensors(buf, device, dc["doc_mask"])
            with net.no_sync() if ddp and i < accum - 1 else nullcontext():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = net(x, y, cu, cu_cpu)
                (loss / accum).backward()
            loss_acc += loss.detach()
            n_acc += 1
        gnorm, bad = apply_grads(model, opt, tc["grad_clip"])
        if bad and master:
            print(f"skip step {step + 1} gnorm {gnorm.item()} top {bad}", flush=True)
        opt.zero_grad(set_to_none=True)
        step += 1

        stop = torch.tensor(float(STOP), device=device)
        if ddp:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        stop = bool(stop.item())

        if step % tc["log_every"] == 0 or step == total or stop:
            avg = loss_acc / max(n_acc, 1)
            if ddp:
                dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            now = time.time()
            tps = tok_step * (step - step_log) / (now - t_log)
            rec = {"step": step, "loss": avg.item(), "lr": lr, "gnorm": gnorm.item(), "tok_s": tps,
                   "tflops_gpu": 6 * n_params * tps / world / 1e12, "mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                   "elapsed_h": (now - t_start) / 3600, "eta_h": (total - step) * tok_step / max(tps, 1) / 3600}
            if master:
                print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            loss_acc.zero_()
            n_acc = 0
            t_log, step_log = now, step

        if step % tc["eval_every"] == 0 or step == total:
            ev = evaluate(model, data, mb, tc["eval_batches"], rank, world, device, dc["doc_mask"])
            if master:
                rec = {"step": step, "val": ev, "val_mean": sum(ev.values()) / len(ev)}
                print("eval", json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            t_log = time.time()

        if step % tc["save_every"] == 0 or step == total or stop:
            if master:
                save(run, model, opt, step, keep=step == total or (tc["keep_every"] > 0 and step % tc["keep_every"] == 0))
                print(f"saved step {step}", flush=True)
            if ddp:
                dist.barrier()
        if stop:
            break

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
