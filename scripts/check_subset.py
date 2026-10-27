"""Check a finished EMG2Pose subset download before building caches on it.

    python scripts/check_subset.py [--subset configs/emg2pose_subset.json]

For every participant of the subset: each ``*_emg.bdf`` has its sidecars
(``_channels.tsv``, ``_emg.json``) and each session its ``_scans.tsv``; then the study,
built with the subset's query, must list exactly the recordings the paper's metadata
(``emg2pose_metadata.csv``, fetched by the download) gives those users, with matching
split / generalization labels. Exits non-zero on any problem.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--subset", type=Path, default=REPO / "configs" / "emg2pose_subset.json")
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())
    config = json.loads(Path(os.environ["NEURALBENCH_CONFIG"]).read_text())
    study_dir = Path(config["DATA_DIR"]) / "Salter2024Emg2pose"
    bids = study_dir / "download" / "nm000281"
    problems = []

    subjects = [s for role in subset["roles"].values() for s in role["subjects"]]
    for subject in subjects:
        folder = bids / f"sub-{subject}"
        bdfs = sorted(folder.glob("ses-*/emg/*_emg.bdf"))
        if not bdfs:
            problems.append(f"sub-{subject}: no recording")
            continue
        for bdf in bdfs:
            for suffix in ("_channels.tsv", "_emg.json"):
                if not bdf.with_name(bdf.name.replace("_emg.bdf", suffix)).exists():
                    problems.append(f"{bdf.name}: no {suffix}")
        for session in folder.glob("ses-*"):
            if not list(session.glob("*_scans.tsv")):
                problems.append(f"sub-{subject}/{session.name}: no scans.tsv")
        if list(folder.rglob("*.part")):
            problems.append(f"sub-{subject}: unfinished .part files")
    print(f"files: {len(subjects)} participants checked, {len(problems)} problems")
    if problems:
        print("\n".join(problems[:30]))
        sys.exit(1)

    from neuralset.events.study import Study

    study = Study(name="Salter2024Emg2pose", path=study_dir, query=subset["query"])
    summary = study.study_summary(apply_query=True)
    meta = pd.read_csv(study_dir / "emg2pose_metadata.csv")
    users = {u for role in subset["roles"].values() for u in role["users"]}
    expected = meta[meta.user.isin(users)]
    print(f"study: {len(summary)} recordings for {summary.subject.nunique()} participants; "
          f"metadata: {len(expected)} recordings for {expected.user.nunique()} users")
    got = Counter(zip(summary.user, summary.split, summary.generalization))
    want = Counter(zip(expected.user, expected.split, expected.generalization))
    if got != want:
        diff = (got - want) + (want - got)
        print(f"MISMATCH on {len(diff)} (user, split, generalization) groups, e.g. {list(diff.items())[:5]}")
        sys.exit(1)
    print(summary.groupby(["split", "generalization"]).size().to_string())
    print("OK")


if __name__ == "__main__":
    os.environ.setdefault("NEURALBENCH_CONFIG", str(REPO / "configs" / "neuralbench.local.json"))
    main()
