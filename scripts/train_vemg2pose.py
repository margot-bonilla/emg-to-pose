"""Train VEMG2Pose on NeuralBench's cached EMG2Pose windows, with the options NeuralBench lacks.

    python scripts/train_vemg2pose.py --out save/train/<name> --subset configs/emg2pose_subset_158.json \
        [--epochs 15] [--seed 33] [--schedule cosine] [--ema 0.999] \
        [--aug-rotation 1] [--aug-gain 0.3] [--aug-noise 0.05] [--norm window] [--all-data] [--resume]

Same data and model as `neuralbench emg pose -m vemg2pose` (the task's windows of 5.895 s with
1,790 samples of left context, scored on the last 5 s; braindecode VEMG2Pose; Adam, L1 loss on
radians), read through the competition loaders (benchmark_utils.nb_task.load_task) from the
cache NeuralBench built. Additions, all off by default:

- augmentation per window: electrode rotation by -k..k (NeuralBench: one offset per batch),
  per-channel gain exp(N(0, s)), Gaussian noise relative to each window's std;
- ``--norm window``: standardise each window per channel inside the model (``PoseNet``), so the
  submission applies it too (``hparams`` in every checkpoint);
- ``--schedule cosine`` (with ``--warmup-epochs``), ``--ema DECAY`` (weights averaged during
  training; checkpoints hold the EMA weights when on);
- ``--all-data``: every user x every stage in train (as scripts/neuralbench_mps.py --all-data);
  no held-out then, val = 2% of recordings for monitoring.

Every epoch: ``OUT/epochs/epoch=NNN.ckpt`` ({"state_dict": model.* weights, "epoch", "hparams"},
readable by scripts/package_*.py and the submissions), a line in ``OUT/metrics.csv`` (train loss,
held-out MAE in degrees = val users x held-out stages, pooled) and ``OUT/last.pt`` (full state for
``--resume``). ``OUT/run_config.json`` records the arguments and commit. Detach long runs with
scripts/detach.py; `make pause-training` sees them.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/emg_pose"
TRAIN_N_TIMES, OUT_TIMES = 11_790, 10_000
DEG = 180 / math.pi


class PoseNet(nn.Module):
    """VEMG2Pose behind an optional per-window, per-channel standardisation."""

    def __init__(self, n_chans, n_joints, sfreq, norm="none"):
        super().__init__()
        from braindecode.models import VEMG2Pose

        self.norm = norm
        self.model = VEMG2Pose(n_chans=n_chans, n_times=TRAIN_N_TIMES, n_outputs=n_joints, sfreq=sfreq)

    def forward(self, X):  # (B, C, T) -> (B, T_out, joints)
        if self.norm == "window":
            X = (X - X.mean(-1, keepdim=True)) / (X.std(-1, keepdim=True) + 1e-6)
        return self.model(X)


def augment(X, rotation, gain, noise, generator):
    """Per-window electrode rotation, per-channel gain and relative Gaussian noise."""
    B, C, _ = X.shape
    if rotation:
        offsets = torch.randint(-rotation, rotation + 1, (B,), generator=generator).to(X.device)
        index = (torch.arange(C, device=X.device)[None, :] - offsets[:, None]) % C
        X = torch.gather(X, 1, index[:, :, None].expand_as(X))
    if gain:
        X = X * torch.exp(gain * torch.randn(B, C, 1, generator=generator)).to(X.device)
    if noise:
        X = X + noise * X.std(-1, keepdim=True) * torch.randn(X.shape, generator=generator).to(X.device)
    return X


def lr_factor(epoch, step, steps_per_epoch, args):
    """Learning-rate multiplier: linear warm-up then constant or cosine to 0."""
    t = epoch + step / steps_per_epoch
    if t < args.warmup_epochs:
        return (t + 1e-3) / args.warmup_epochs
    if args.schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * (t - args.warmup_epochs) / max(1e-9, args.epochs - args.warmup_epochs)))
    return 1.0


@torch.inference_mode()
def heldout_mae(net, loader, device, limit=None):
    """Pooled MAE in degrees on the last 5 s of each window (real left context)."""
    net.eval()
    total, count = 0.0, 0
    for i, (X, y, _) in enumerate(loader):
        if limit and i >= limit:
            break
        pred = net(X.to(device).float()).transpose(1, 2)
        total += float((pred - y[..., -OUT_TIMES:].to(device)).abs().sum())
        count += pred.numel()
    net.train()
    return total / max(count, 1) * DEG


def git_commit():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset_158.json")
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay per step, e.g. 0.999 (0 = off)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--aug-rotation", type=int, default=0, help="max electrode offset per window")
    parser.add_argument("--aug-batch-rotation", action="store_true",
                        help="NeuralBench's augmentation: one offset in -1..1 per batch")
    parser.add_argument("--aug-gain", type=float, default=0.0, help="std of the log per-channel gain")
    parser.add_argument("--aug-noise", type=float, default=0.0, help="noise std relative to the window std")
    parser.add_argument("--norm", choices=["none", "window"], default="none")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None, help="smoke test: batches per epoch and for monitoring")
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_utils.nb_task import load_task
    from eval_local import heldout_loader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    config = json.loads(Path(os.environ["NEURALBENCH_CONFIG"]).read_text())
    cache = config["CACHE_DIR"]
    overrides = {
        "duration": TRAIN_N_TIMES / 2000.0,  # VEMG2Pose's windows (neuralbench/models/vemg2pose.yaml)
        "study.source.query": json.loads(args.subset.read_text())["query"],
        "study.source.infra.folder": cache, "neuro.infra.folder": cache,
        "target.infra.folder": cache, "target.infra.cluster": None, "target.infra.permissions": None,
    }
    if args.all_data:
        overrides.update({
            "study.keep_held_out_user_stage_test.query": "duration >= 0",
            "study.split.test_split_query": "split == 'test' and generalization == 'user_stage'",
            "study.split.valid_split_by": "timeline",
            "study.split.valid_split_ratio": 0.02,
        })
    loaders, meta = load_task("emg", "pose", data_dir=config["DATA_DIR"], device="cpu",
                              batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers,
                              overrides=overrides)
    n_joints = int(meta["target_shape"][0])
    train_loader = loaders["train"]
    monitor = loaders["val"] if args.all_data else heldout_loader(loaders["val"], 128)
    print(f"train {len(train_loader.dataset)} windows, monitor ({'val 2%' if args.all_data else 'held-out'}) "
          f"{len(monitor.dataset)} windows, device {device}", flush=True)

    hparams = {"norm": args.norm, "n_chans": meta["n_chans"], "n_joints": n_joints, "sfreq": meta["sfreq"]}
    net = PoseNet(meta["n_chans"], n_joints, meta["sfreq"], args.norm).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    ema = None
    if args.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema))

    out = args.out
    (out / "epochs").mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and (out / "last.pt").exists():
        # On the CPU: RNG states must stay there; load_state_dict moves weights to their params' device.
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        optimizer.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start = state["epoch"] + 1
        print(f"resumed after epoch {state['epoch']}", flush=True)
    else:
        (out / "run_config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "commit": git_commit(), "command": " ".join(sys.argv), "device": device}, indent=1) + "\n")

    metrics_path = out / "metrics.csv"
    new_metrics = not metrics_path.exists() or start == 0
    with open(metrics_path, "w" if new_metrics else "a", newline="") as f:
        if new_metrics:
            csv.writer(f).writerow(["epoch", "train_l1_deg", "monitor_mae_deg", "lr", "seconds"])

    steps = len(train_loader)
    loss_fn = nn.L1Loss()
    for epoch in range(start, args.epochs):
        t0, running, n = time.perf_counter(), 0.0, 0
        net.train()
        for step, (X, y, _) in enumerate(train_loader):
            if args.limit_batches and step >= args.limit_batches:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_factor(epoch, step, steps, args)
            X, y = X.to(device).float(), y[..., -OUT_TIMES:].to(device)
            if args.aug_batch_rotation:
                X = torch.roll(X, int(torch.randint(-1, 2, (1,), generator=generator)), dims=1)
            X = augment(X, args.aug_rotation, args.aug_gain, args.aug_noise, generator)
            loss = loss_fn(net(X).transpose(1, 2), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update_parameters(net)
            running, n = running + float(loss), n + 1
        scored = ema.module if ema is not None else net
        mae = heldout_mae(scored, monitor, device, args.limit_batches)
        seconds = time.perf_counter() - t0
        torch.save({"state_dict": {f"model.{k}": v for k, v in scored.model.state_dict().items()},
                    "epoch": epoch, "hparams": hparams}, out / "epochs" / f"epoch={epoch:03d}.ckpt")
        torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "epoch": epoch,
                    "generator": generator.get_state(), "torch_rng": torch.get_rng_state()}, out / "last.pt")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{running / n * DEG:.4f}", f"{mae:.4f}",
                                    f"{optimizer.param_groups[0]['lr']:.2e}", f"{seconds:.0f}"])
        print(f"[epoch {epoch}] {seconds:.0f} s, train L1 {running / n * DEG:.3f} deg, "
              f"{'val' if args.all_data else 'held-out'} MAE {mae:.3f} deg", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("NEURALBENCH_CONFIG", str(REPO / "configs" / "neuralbench.local.json"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
