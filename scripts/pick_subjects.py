"""Pick the EMG2Pose participants to download, and write configs/emg2pose_subset.json.

    python -I scripts/pick_subjects.py --metadata emg2pose_metadata.csv \
        --participants participants.tsv [--n-train 40] [--seed 0]

Inputs (small, fetched once in a terminal):
- https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv (the paper's splits)
- https://raw.githubusercontent.com/nemardatasets/nm000281/main/participants.tsv
  (BIDS label -> the metadata's anonymized ``user``)

Keeps every test user (their user_stage recordings are the warm-up set, so local
scores can match Codabench) and every val user (val user_stage = unseen users x
unseen stages: what we select on and early-stop on), plus ``--n-train`` train users
drawn at random. Sizes are estimated at 0.80 GB per recorded hour (340 GB / 424 h).

The output lists BIDS labels per role and the study query, ``subject in [...]``,
used by scripts/download_subset.py, scripts/neuralbench_mps.py and scripts/eval_local.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GB_PER_HOUR = 0.80


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    participants = pd.read_csv(args.participants, sep="\t")
    label_of = {user: pid.removeprefix("sub-")
                for pid, user in zip(participants.participant_id, participants.original_user)}
    hours = ((meta.end - meta.start) / 3600).groupby(meta.user).sum()

    # Train users also have test rows (generalization == stage: their held-out
    # stages); val and test users only have rows of their own split.
    role_of = meta.groupby("user").split.agg(
        lambda s: "train" if "train" in set(s) else "val" if "val" in set(s) else "test")
    users = {role: sorted(role_of.index[role_of == role]) for role in ("train", "val", "test")}
    rng = np.random.default_rng(args.seed)
    users["train"] = sorted(rng.choice(users["train"], size=args.n_train, replace=False))

    roles = {}
    for role in ("test", "val", "train"):
        roles[role] = {
            "users": users[role],
            "subjects": [label_of[u] for u in users[role]],
            "est_gb": round(float(hours[users[role]].sum() * GB_PER_HOUR), 1),
        }
    # The study's events label subjects "<Study>/<label>"; its downloader strips the prefix.
    subjects = sorted(f"Salter2024Emg2pose/{s}" for r in roles.values() for s in r["subjects"])
    out = {
        "description": f"all test + all val users, {args.n_train} random train users (seed {args.seed})",
        "seed": args.seed,
        "roles": roles,
        "est_gb": round(sum(r["est_gb"] for r in roles.values()), 1),
        "query": f"subject in {subjects!r}",
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for role, r in roles.items():
        print(f"{role:>5}: {len(r['subjects']):3d} subjects, ~{r['est_gb']} GB")
    print(f"total: {len(subjects)} subjects, ~{out['est_gb']} GB -> {args.out}")


if __name__ == "__main__":
    main()
