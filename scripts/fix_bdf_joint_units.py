"""Fix EMG2Pose BDF files whose joint angles read 10^6 too large.

    python -I scripts/fix_bdf_joint_units.py data/neural_compet/Salter2024Emg2pose/download/nm000281 [--dry-run]

Every NEMAR nm000281 recording stores its 20 joint-angle channels in micro-radians
(header physical range about -8e5..2e6). Older files label them ``uV``, so MNE divides
by 1e6 and the angles read in radians, as NeuralBench, VEMG2Pose and the Codabench
evaluation expect (the MeanPose floor scores 22.63 deg there). Files re-uploaded since
(seen on 2026-10-27: 94% of our subset, sometimes mixed within a subject) label them
``rad`` without rescaling the samples, so they read 1e6 too large.

This rewrites the 8-byte physical-dimension field of those joint channels from ``rad``
to ``uV`` (header only; samples untouched), so every file reads like the old ones.
A channel is changed only if it is a ``joint*`` channel labelled ``rad`` whose header
physical range exceeds 1000 (micro-units). Each change is appended to
``joint_unit_fix.csv`` next to the files (to undo: write ``rad`` back). Idempotent.
Then delete the target caches built from the bad files.
"""
import argparse
import csv
import glob
from pathlib import Path

LIMIT = 1000.0


def header_fields(f):
    """Labels, physical dims and ranges of a BDF/EDF, plus each dim field's file offset."""
    f.seek(0)
    head = f.read(256)
    n = int(head[252:256])
    signal_head = f.read(256 * n)

    def field(offset, width, i):
        start = offset * n + width * i
        return signal_head[start:start + width].decode("ascii", "replace").strip(), 256 + start

    # Per-signal header layout: label 16, transducer 80, dimension 8, phys min 8, phys max 8, ...
    out = []
    for i in range(n):
        label, _ = field(0, 16, i)
        dim, dim_pos = field(16 + 80, 8, i)
        pmin, _ = field(16 + 80 + 8, 8, i)
        pmax, _ = field(16 + 80 + 16, 8, i)
        out.append(dict(label=label, dim=dim, dim_pos=dim_pos, pmin=float(pmin), pmax=float(pmax)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = args.bids_root / "joint_unit_fix.csv"
    files = sorted(glob.glob(str(args.bids_root / "sub-*/ses-*/emg/*_emg.bdf")))
    counts = {"fixed": 0, "already uV": 0, "skipped (radian range)": 0}
    rows = []
    for path in files:
        with open(path, "rb" if args.dry_run else "r+b") as f:
            joints = [s for s in header_fields(f) if s["label"].startswith("joint")]
            to_fix = [s for s in joints if s["dim"] == "rad" and max(abs(s["pmin"]), abs(s["pmax"])) > LIMIT]
            if not to_fix:
                key = "already uV" if all(s["dim"] == "uV" for s in joints) else "skipped (radian range)"
                counts[key] += 1
                continue
            for s in to_fix:
                if not args.dry_run:
                    f.seek(s["dim_pos"])
                    f.write(b"uV".ljust(8))
                rows.append(dict(file=str(Path(path).relative_to(args.bids_root)), channel=s["label"],
                                 old_dim="rad", new_dim="uV", pmin=s["pmin"], pmax=s["pmax"]))
            counts["fixed"] += 1
        if not args.dry_run:
            with open(path, "rb") as f:  # read back
                assert all(s["dim"] == "uV" for s in header_fields(f) if s["label"] in {r["label"] for r in to_fix})

    if rows and not args.dry_run:
        new = not manifest.exists()
        with open(manifest, "a", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
    print(f"{len(files)} files: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + (" (dry run)" if args.dry_run else f"; {len(rows)} channels logged in {manifest}"))


if __name__ == "__main__":
    main()
