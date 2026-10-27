"""What the official VEMG2Pose template loses by inventing its left context.

    python scripts/diag_left_context.py submissions/<vemg2pose folder> [--split heldout]

NeuralBench trains VEMG2Pose on 5.895 s windows: 1,790 samples (0.895 s) of left context,
predictions for the last 5 s. Codabench gives bare 5 s windows, so the template pads them
by reflection. On the same held-out windows taken with their real context, this scores the
same weights on the same last 5 s of target, (a) with the real context and (b) with the
template's padding of the bare 5 s. MAE per subject, in degrees, like the Codabench Score.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/emg_pose"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submission", type=Path)
    parser.add_argument("--split", default="heldout", choices=["heldout", "val"])
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_utils.nb_task import load_task
    from eval_local import heldout_loader, load_model

    config = json.loads(Path(os.environ["NEURALBENCH_CONFIG"]).read_text())
    cache = config["CACHE_DIR"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    loaders, meta = load_task(
        "emg", "pose", data_dir=config["DATA_DIR"], device=device, batch_size=32,
        overrides={
            "duration": 5.895,  # VEMG2Pose's training windows (models/vemg2pose.yaml)
            "study.source.query": json.loads(args.subset.read_text())["query"],
            "study.source.infra.folder": cache, "neuro.infra.folder": cache,
            "target.infra.folder": cache, "target.infra.cluster": None, "target.infra.permissions": None,
        },
    )
    n_joints = int(meta.pop("target_shape")[0])
    meta.pop("raw_target_shape", None)
    loader = heldout_loader(loaders["val"], 32) if args.split == "heldout" else loaders["val"]
    model = load_model(args.submission, {**meta, "n_joints": n_joints}, device, "submission")

    errors = {"real context": {}, "template padding": {}}
    n_times = None
    with torch.inference_mode():
        for X, y, info in loader:
            target = y[..., -10_000:]
            n_times = X.shape[-1]
            preds = {
                "real context": model.net(X.float()).transpose(1, 2),
                "template padding": model.predict(X[..., -10_000:]),
            }
            subjects = np.asarray(info["subject_id"]).reshape(-1)
            for name, pred in preds.items():
                err = (pred - target).abs().nanmean(dim=(1, 2)).cpu().numpy()
                for s, e in zip(subjects, err):
                    errors[name].setdefault(int(s), []).append(float(e))
    print(f"{args.split}: {len(loader.dataset)} windows of {n_times} samples, "
          f"scored on the last 10,000")
    for name, per_subject in errors.items():
        means = [np.mean(v) for v in per_subject.values()]
        print(f"  {name:>16}: {np.degrees(np.mean(means)):.3f} deg (per-subject mean, {len(means)} subjects)")


if __name__ == "__main__":
    os.environ.setdefault("NEURALBENCH_CONFIG", str(REPO / "configs" / "neuralbench.local.json"))
    main()
