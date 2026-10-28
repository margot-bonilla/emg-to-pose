"""Package several NeuralBench VEMG2Pose checkpoints as one ensemble submission folder.

    python scripts/package_ensemble.py submissions/<name> CKPT [CKPT ...] \
        [--subset configs/emg2pose_subset_100.json] [--command "..."] [--selection "..."]

Writes ``submission.py`` (submission/vemg2pose_ensemble_submission.py: mean of the members'
predicted angles, same model and padding as the official template), ``weights_00.pt``, ...
(each member's ``model.*`` weights) and ``run_config.json`` (members, commits, command; not
zipped). Then: ``make test-submission SUBMISSION=<folder>`` and ``scripts/eval_local.py <folder>``.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from package_vemg2pose import COMPET, REPO, git_commit, relative

ENSEMBLE = REPO / "submission" / "vemg2pose_ensemble_submission.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    parser.add_argument("--command", default=None, help="training command(s), recorded in run_config.json")
    parser.add_argument("--selection", default=None, help="how the members were chosen")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("weights*.pt"):
        old.unlink()
    shutil.copyfile(ENSEMBLE, args.out / "submission.py")
    members = []
    for i, path in enumerate(args.checkpoints):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in checkpoint.get("state_dict", checkpoint).items() if k.startswith("model.")}
        if not state:
            raise SystemExit(f"no model.* weights in {path}")
        torch.save({"state_dict": state}, args.out / f"weights_{i:02d}.pt")
        members.append({"file": f"weights_{i:02d}.pt", "checkpoint": relative(path), "epoch": checkpoint.get("epoch")})
    run_config = {
        "members": members,
        "repo_commit": git_commit(REPO),
        "competition_commit": git_commit(COMPET),
        "subset": relative(args.subset),
        "subset_sha1": hashlib.sha1(args.subset.read_bytes()).hexdigest(),
        "command": args.command,
        "selection": args.selection,
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=1) + "\n")
    size = sum(p.stat().st_size for p in args.out.glob("weights_*.pt")) / 1e6
    print(f"{args.out}: submission.py + {len(members)} weights ({size:.0f} MB), epochs {[m['epoch'] for m in members]}")


if __name__ == "__main__":
    main()
