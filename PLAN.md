# Track 4 plan (written Oct 22, 2026)

Goal: 2–3 well-tested candidates ready by **Nov 4**, selected on a held-out split of
**unseen users × unseen stages** that mirrors the sealed test, never on the warm-up test set.

## Hard dates

| Date | What |
|---|---|
| **as soon as possible, before Sun Nov 1, 23:59 CET** | Everyone registers for **Track 4** on Codabench (separate from Track 1) and joins the fireflies organisation |
| Nov 1, 23:59 CET | Warm-up ends: last chance for free uploads to check local = Codabench |
| Nov 2–4 | No uploads: final retrains and contract checks |
| Nov 5 → Nov 21, 23:59 CET | Sealed phase: 5 uploads per person in total |

## Constraints

- **GPU:** Track 1 has priority on Margot's Mac, especially at night. Run `make training-status`
  (lists every run on the Mac) before any GPU job; never run two trainings at once.
- **No dependency on Track 1:** own config, data and caches (duplicated if needed); files are copied, never linked.
- **Workflow:** push directly to `main` until the structure is settled and the first end-to-end
  (milestone 3) works; pull requests after that (merge commits only, for the audit). Heavy Track 4 training belongs on Enrique's Mac or in daytime gaps.
- **Downloads:** Margot runs them in her own terminal.
- **Disk:** 535 GB free (Oct 22), shared with Track 1's caches.

## Milestones

| # | When | Goal | Done when |
|---|---|---|---|
| 1 | Oct 23 | `make env`, official repo, `make starter` | `starter.yml` (MeanPose on Simulated) passes |
| 2 | Oct 23–24 | EMG2Pose: **check size on NEMAR (NM000281)**, decide where `data/` lives, download, `--prepare` | the `emg/pose` task loads a train and a test batch |
| 3 | Oct 24–25 | `scripts/eval_local.py` (scores folders like Codabench, plus ensembles) + a first baseline (`Torch-Linear`, then NeuralBench VEMG2Pose) | one warm-up upload where local MAE = Codabench MAE; then delete `docs/handoff_track4_emg_pose.md` (its useful parts live in README/PLAN) |
| 4 | Oct 25–26 | Held-out split committed in `configs/` | every run reports a held-out MAE |
| 5 | Oct 26–28 | Window cache, `scripts/train_solver.py`, `submission/submission.py` with per-epoch checkpoints, `--resume` and smoothed early stopping | a restartable run that exports an upload-ready folder with `run_config.json` |
| 6 | Oct 28 – Nov 1 | Experiments on held-out (list below) | a recipe that clearly beats the baseline on held-out |
| 7 | Nov 2–4 | Final retrains on all users (fixed epoch count), seed ensembles, `make test-submission` | 2–3 candidates in `submissions/`, logged in `RESULTS.md` |

Sealed phase: upload 2–3 very different candidates early, refine what wins, end with the best.

## Held-out split (milestone 4)

EMG2Pose has 193 users and 29 stages. Pick ~10–15% of users **and** a few stages as held-out:
train on (train users × train stages), validate on (held-out users × held-out stages). The other two
combinations are left out during development and come back for the final retrain. Check that the
held-out stages cover different kinds of movement, not one family. Commit the split as JSON (like
Track 1's `configs/heldout_*.json`) so every run uses the same one.

## Experiments (milestone 6), in rough order of expected value

1. **Train on 5 s windows without left context.** NeuralBench's VEMG2Pose trains on 5.895 s windows
   (1,790 samples of left context); Codabench gives exactly 5 s, so the official template pads by
   reflection. Training on the evaluated format should remove that train/test mismatch.
2. **Normalisation:** per-window and per-channel normalisation of the raw µV (amplitude depends on
   skin and band placement), against the pipeline's default.
3. **Band-rotation augmentation:** roll the 16 channels by ±1 electrode (the wristband sits
   differently on each person), plus amplitude scaling per channel.
4. **Number of participants:** train on 25 / 50 / 100 / all train users; on Track 1, more people
   was the biggest lever for unseen people.
5. **Model size and loss:** L1 on angles in radians (the metric) vs smooth L1 / MSE; a wider or deeper network.
6. **Pretrained models:** check whether NeuralBench's Track 4 baselines fine-tune anything
   (`neuralbench/defaults/downstream_wrappers.yaml`); if so, fine-tune fully, don't just probe.
7. **Seed ensembles** (average of angles) for the final candidates: +2–3 points on Track 1, and steadier in the audit.

Early stopping and selection always use a **smoothed** held-out score (mean of the last 3 epochs, patience ~10).

## To copy from Track 1 (`../egg-to-image`) as the milestones need them

| Milestone | Files | Changes for Track 4 |
|---|---|---|
| 3 | `scripts/eval_local.py` | key `angular_mae`, lower is better, ensemble = mean of angles |
| 5 | `src/data/window_cache.py`, `scripts/cache_windows.py` | targets are `(20, T)` per window; consider float16 (640 KB per window in float32) |
| 5 | `scripts/train_solver.py`, `submission/submission.py` (`Solver.fit`) | objective `EMG-pose`, minimise MAE, regression loss instead of CLIP |
| 5 | `scripts/neuralbench_mps.py`, `scripts/mps_patch.py` | Lightning on `mps` for the NeuralBench start kit |
| any | `scripts/docs_to_latex.py` | none |

Already here: `Makefile`, `scripts/neuralbench_config.py` (now a repo-local config),
`scripts/training.sh` (machine-wide, labels each run's project), `scripts/battery_guard.sh`, `RESULTS.md`.

## Open questions

- EMG2Pose download size (it lives in this repo's `data/`, not shared with Track 1).
- One copy on disk: the NeuralBench CLI downloads to `data/`, while benchopt's `Salter2024Emg2pose`
  uses its own `get_data_path("neural_compet")` (by default inside `external/`). Point both at the
  same folder before downloading (e.g. `BENCHOPT_DATA_HOME`), or the data is fetched twice.
- Training cost of NeuralBench's VEMG2Pose on a Mac (`mps`).
- Whether anything in the Track 4 baselines is pretrained.
- Who uploads what in the sealed phase (5 uploads each, separate from Track 1's).
- Whether Enrique's Mac is available for long runs, and how to sync data/checkpoints with it.
