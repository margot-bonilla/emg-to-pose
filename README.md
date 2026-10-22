# Neural Interfaces 2026 - Track 4: EMG-to-Pose

Team **fireflies** · Official repo: [neural-interfaces26/2026-competition](https://github.com/neural-interfaces26/2026-competition) (`tracks/emg_pose/`) ·
[NeuralBench Track 4 guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html) ·
Workshop: NeurIPS 2026 Brain & Body Workshop

Start here: [`docs/handoff_track4_emg_pose.md`](docs/handoff_track4_emg_pose.md) (rules, contract, lessons from Track 1), then [`PLAN.md`](PLAN.md).

---

## 🎯 The task

From **16-channel wrist surface EMG**, predict the trajectories of **20 UmeTrack hand-joint angles**.

- **Input:** `X` of shape `(B, 16, 10000)`: independent, non-overlapping **5 s windows at 2 kHz**, **raw** (only volts → µV). No state across `predict` calls; any batch size and order.
- **Output:** `(B, n_joints, T')` **in radians**, `n_joints = meta["n_joints"] = 20`. A coarser `T'` is nearest-resampled to the target length. Read every size from `meta`.
- **Metric:** **mean absolute angular error in degrees** over examples × joints × time (lower is better). Reference: NeuroPose ≈ 17.5° ± 1.5.
- **Generalization:** the sealed test is a private 2026 Meta cohort with **new users and new movement stages**.

| | Warm-up | Sealed |
|---|---|---|
| Data | EMG2Pose `user_stage` test split (public labels) | private 2026 Meta Reality Labs cohort |
| Dates | until **Sun Nov 1, 23:59 CET** (registration closes too) | **Nov 5 00:01 → Nov 21 23:59 CET** |
| Uploads | free, daily quota | **5 per person, in total** |
| Counts for the prize? | no | yes; top 3 get a reproducibility audit (±2σ) |

## 🚀 Setup

```bash
make env                   # Python 3.12 .venv + requirements + official repo in external/ + NeuralBench config
source .venv/bin/activate
make starter               # zero-download contract check (MeanPose on Simulated data)
make training-status       # before ANY GPU job: all runs on this Mac (Track 1 has priority)
```

- **Python 3.12 only**, like the Codabench worker; `python3.12` must be on PATH. `requirements.txt` pins `neuralbench`/`neuralset`/`neuralfetch`/`neuraltrain==0.4.0` and `exca==0.5.29`, the worker's versions. Don't change these pins.
- **Independent of other projects:** NeuralBench uses this repo's own `configs/neuralbench.local.json` (this repo's `data/`, `cache/`, `save/`), selected by `NEURALBENCH_CONFIG`, which the Makefile and `.venv/bin/activate` export. The global `~/.neuralbench/config.json` is never used.
- `make pause-training` / `resume-training` act on **every** training run on the Mac, Track 1's included: pause before closing the laptop.
- **Downloads run in your own terminal**, not in Claude's sandbox. Check the dataset size first.

## 📁 Layout

```
Makefile              env, contract checks, packaging, training pause/resume
requirements.txt      pinned to the Codabench worker stack
PLAN.md               plan and milestones up to the sealed phase
RESULTS.md            one row per upload (commit, command, local and Codabench scores)
docs/                 handoff and write-ups
scripts/              setup, training and evaluation tools (train_solver.py, eval_local.py, ...)
src/data/             window cache, held-out split
src/models/           model code shared by the submissions
submission/           the current submission.py (+ weights)
submissions/<name>/   one folder per uploaded candidate (ZIP content + run_config.json)
configs/              committed held-out split and run configs
external/             official competition repo (gitignored)
data/ cache/ save/    datasets, caches, NeuralBench outputs (gitignored)
```
