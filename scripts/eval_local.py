"""Score submission folders locally, exactly like Codabench, on the real EMG2Pose data.

    python scripts/eval_local.py submissions/a [submissions/b ...] [--split heldout]
    python scripts/eval_local.py submissions/vemg2pose --epochs save/<run>/epochs --split heldout

Adapted from ../egg-to-image/scripts/eval_local.py (copied, not shared). Loads each
folder's submission.py + weights the way the platform does (``Solver.load_model(meta)``),
feeds it the competition's own data pipeline (benchmark_utils.nb_task.load_task) and
scores it with the official objective (tracks/emg_pose/objective.py). Key metric:
``angular_mae_subject_mean`` in degrees (mean over subjects of each one's MAE), **lower is
better**: it is the Score Codabench shows (MeanPose floor: 22.6283 here, 22.63 there), not
the pooled ``angular_mae`` (22.6666). With several folders it also scores
their ensemble (mean of the predicted angles).

Splits:
- ``test``: the paper's test users x held-out stages (generalization == user_stage),
  i.e. the warm-up evaluation set. Report only, never select on it.
- ``heldout``: val users x held-out stages (val, generalization == user_stage): unseen
  users AND unseen stages, like the sealed test. Select and early-stop on this.
- ``val``: the whole val split (both val scenarios), what NeuralBench's ``val/mae`` pools.

``--epochs DIR`` scores every ``*.ckpt`` in DIR (scripts/neuralbench_mps.py writes one per
epoch) with the first folder's submission.py, and prints the score smoothed over the
last 3 epochs: pick the epoch with the best smoothed score.

Data: only the participants of ``--subset`` (configs/emg2pose_subset.json), from
NeuralBench's DATA_DIR and CACHE_DIR (configs/neuralbench.local.json), so nothing is
downloaded or extracted twice.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/emg_pose"
METRICS = ["angular_mae_subject_mean", "angular_mae", "angular_mae_subject_std", "angular_rmse",
           "r2_score", "n_subjects"]
SMOOTH = 3


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ensemble:
    """Mean of the members' predicted angles (radians)."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [torch.as_tensor(m.predict(X)).float() for m in self.models]
        return torch.stack(preds).mean(0)


def load_model(folder, meta, device, name, weights=None):
    """``Solver.load_model`` from ``folder``; ``weights`` replaces its weights.pt."""
    submission = load_module(name, folder / "submission.py")
    solver = submission.Solver.get_instance()
    solver.train_loader = None  # inference-only, like the platform
    if weights is None:
        return solver.load_model({**meta, "device": device, "submission_dir": folder.resolve()})
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "weights.pt").symlink_to(Path(weights).resolve())
        return solver.load_model({**meta, "device": device, "submission_dir": Path(tmp)})


def heldout_loader(val_loader, batch_size):
    """The val windows of held-out stages (generalization == user_stage)."""
    from torch.utils.data import DataLoader, Subset

    from benchmark_utils.nb_task import _DeviceLoader

    windows = val_loader.dataset  # nb_task._NBWindows
    segments = windows.seg_ds.segments
    keep = [i for i, j in enumerate(windows.index)
            if segments[int(j)].trigger.to_dict().get("generalization") == "user_stage"]
    if not keep:
        raise RuntimeError("no val window with generalization == 'user_stage'")
    loader = DataLoader(Subset(windows, keep), batch_size=batch_size, shuffle=False)
    return _DeviceLoader(loader, val_loader.device)


def score(objective, loader, model, meta):
    objective.set_data(train_loader=None, test_loader=loader, **meta)  # meta has n_joints
    return objective.evaluate_result(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submissions", nargs="+", type=Path, help="folders with submission.py + weights")
    parser.add_argument("--split", default="test", choices=["test", "heldout", "val"])
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    parser.add_argument("--epochs", type=Path, default=None, metavar="DIR",
                        help="score every *.ckpt in DIR with the first folder's submission.py")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
    from benchmark_utils.nb_task import load_task

    subset = json.loads(args.subset.read_text())
    query = subset["query"]
    if args.split == "test":
        # The test users alone, plus one val and one train user: NeuralBench 0.4.0's
        # Data.prepare refuses an empty split (so nb_task's subset="test" fails), and
        # this avoids extracting the whole subset. The test windows don't depend on them.
        roles = subset["roles"]
        labels = roles["test"]["subjects"] + roles["val"]["subjects"][:1] + roles["train"]["subjects"][:1]
        query = f"subject in {[f'Salter2024Emg2pose/{s}' for s in labels]!r}"
    config = json.loads(Path(os.environ["NEURALBENCH_CONFIG"]).read_text())
    cache = config["CACHE_DIR"]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    loaders, meta = load_task(
        "emg", "pose", data_dir=config["DATA_DIR"], device=device, batch_size=args.batch_size,
        num_workers=args.num_workers,
        overrides={
            "study.source.query": query,
            "study.source.infra.folder": cache,
            "neuro.infra.folder": cache,
            "target.infra.folder": cache,
            "target.infra.cluster": None,
            "target.infra.permissions": None,
        },
    )
    n_joints = int(meta.pop("target_shape")[0])
    meta.pop("raw_target_shape", None)
    meta = {**meta, "n_joints": n_joints}
    loader = {"test": lambda: loaders["test"], "val": lambda: loaders["val"],
              "heldout": lambda: heldout_loader(loaders["val"], args.batch_size)}[args.split]()

    objective = load_module("objective", TRACK / "objective.py").Objective.get_instance()

    models = {}
    if args.epochs:
        for ckpt in sorted(args.epochs.glob("*.ckpt")):
            models[ckpt.stem] = load_model(args.submissions[0], meta, device, f"sub_{ckpt.stem}", ckpt)
    else:
        for i, folder in enumerate(args.submissions):
            models[folder.name] = load_model(folder, meta, device, f"submission_{i}")
        if len(models) > 1:
            models[f"ensemble of {len(models)}"] = Ensemble(list(models.values()))

    print(f"\n{args.split} split: {len(loader.dataset)} windows, {meta['n_chans']} channels x "
          f"{meta['n_times']} samples, device={device}\n")
    width = max(len(name) for name in models) + 2
    columns = METRICS + ([f"mae_smoothed{SMOOTH}"] if args.epochs else [])
    print(f"{'model':<{width}}" + "".join(f"{m:>26}" for m in columns))
    history = []
    for name, model in models.items():
        result = score(objective, loader, model, meta)
        history.append(result["angular_mae_subject_mean"])
        row = [f"{result[m]:>26.4f}" if m != "n_subjects" else f"{int(result[m]):>26d}" for m in METRICS]
        if args.epochs:
            row.append(f"{np.mean(history[-SMOOTH:]):>26.4f}")
        print(f"{name:<{width}}" + "".join(row), flush=True)
    if args.epochs and history:
        smoothed = [np.mean(history[max(0, i - SMOOTH + 1):i + 1]) for i in range(len(history))]
        best = int(np.argmin(smoothed))
        print(f"\nbest smoothed ({SMOOTH} epochs): {list(models)[best]}, {smoothed[best]:.4f} deg; "
              f"best single: {list(models)[int(np.argmin(history))]}, {min(history):.4f} deg")


if __name__ == "__main__":
    os.environ.setdefault("NEURALBENCH_CONFIG", str(REPO / "configs" / "neuralbench.local.json"))
    main()
