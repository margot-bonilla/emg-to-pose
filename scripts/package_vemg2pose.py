"""Package a NeuralBench VEMG2Pose checkpoint as a submission folder.

    python scripts/package_vemg2pose.py save/<run>/epochs/epoch=041.ckpt submissions/vemg2pose_s75 \
        --command "python scripts/neuralbench_mps.py emg pose -m vemg2pose --subset ... --seeds 33"

Writes into the folder:
- ``submission.py``: the official template
  (tracks/emg_pose/submission_templates/neuralbench_vemg2pose_submission.py), unchanged;
- ``weights.pt``: the brain model's weights only (``model.*`` keys of the Lightning checkpoint);
- ``run_config.json``: checkpoint, epoch, commits and command, for the audit (not zipped).

Then: ``make test-submission SUBMISSION=<folder>`` and ``scripts/eval_local.py <folder>``.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMPET = REPO / "external/2026-competition"
TEMPLATE = COMPET / "tracks/emg_pose/submission_templates/neuralbench_vemg2pose_submission.py"


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def git_commit(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--command", default=None, help="training command, recorded in run_config.json")
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    model_state = {k: v for k, v in state.items() if k.startswith("model.")}
    if not model_state:
        raise SystemExit(f"no model.* weights in {args.checkpoint}")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, args.out / "submission.py")
    if (checkpoint.get("hparams") or {}).get("norm", "none") != "none":
        raise SystemExit("this checkpoint normalises windows: package it with scripts/package_ensemble.py")
    torch.save({"state_dict": model_state}, args.out / "weights.pt")
    run_config = {
        "checkpoint": relative(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "repo_commit": git_commit(REPO),
        "competition_commit": git_commit(COMPET),
        "subset": relative(args.subset),
        "subset_sha1": hashlib.sha1(args.subset.read_bytes()).hexdigest(),
        "command": args.command,
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=1) + "\n")
    print(f"{args.out}: submission.py + weights.pt "
          f"({(args.out / 'weights.pt').stat().st_size / 1e6:.1f} MB), epoch {run_config['epoch']}")


if __name__ == "__main__":
    main()
