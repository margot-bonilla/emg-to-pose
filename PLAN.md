# Track 4 plan (written Oct 22, 2026; updated Oct 27)

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
- **Disk:** 999 GB free (Oct 27), shared with Track 1's caches.

## Milestones

| # | When | Goal | Done when |
|---|---|---|---|
| 1 ✅ | Oct 22 | `make env`, official repo, `make starter` | `starter.yml` (MeanPose on Simulated) passes: 20.8° on Simulated |
| 2 | Oct 27–28 | EMG2Pose subset (75 participants, ~132 GB, see below): `make download-subset`, then `--prepare` | the `emg/pose` task loads a train and a test batch |
| 3 | Oct 27–Nov 1 | `scripts/eval_local.py` ✅ (written, untested until data lands) + NeuralBench VEMG2Pose on the subset (`scripts/neuralbench_mps.py`, `scripts/package_vemg2pose.py` ✅) | one warm-up upload where local MAE = Codabench MAE; then delete `docs/handoff_track4_emg_pose.md` (its useful parts live in README/PLAN) |
| 4 ✅ | Oct 27 | Held-out split = the paper's val users × held-out stages (`eval_local.py --split heldout`) | every run reports a held-out MAE |
| 5 | Oct 26–28 | Window cache, `scripts/train_solver.py`, `submission/submission.py` with per-epoch checkpoints, `--resume` and smoothed early stopping | a restartable run that exports an upload-ready folder with `run_config.json` |
| 6 | Oct 28 – Nov 1 | Experiments on held-out (list below) | a recipe that clearly beats the baseline on held-out |
| 7 | Nov 2–4 | Final retrains on all users (fixed epoch count), seed ensembles, `make test-submission` | 2–3 candidates in `submissions/`, logged in `RESULTS.md` |

Sealed phase: upload 2–3 very different candidates early, refine what wins, end with the best.

## Warm-up uploads

- Oct 27: `dist/meanpose_floor.zip` (official MeanPose verbatim; with no weights it predicts the all-zero pose). Only checks the registration and the upload path.
- Next: VEMG2Pose on the subset, once `eval_local.py --split test` gives its local score (it must match Codabench).

## Held-out split (milestone 4)

**Decision (Oct 27): use the paper's own split.** Its 15 val users are held out from training, and their
`generalization == user_stage` recordings (330 recordings, ~3.7k windows) cover the 6 held-out stages, which
no training user is trained on (the train users' recordings of those stages are in `test/stage`, dropped by
the task). So val `user_stage` = unseen users × unseen stages: select and early-stop on it
(`scripts/eval_local.py --split heldout`). Caveat: those 6 stages are the same as in the warm-up test, while
the sealed test has new stages. The text below was the original idea, kept for the final retrain.


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
   skin and band placement), against the pipeline's default (none: raw µV into VEMG2Pose's LayerNorm).
3. **Augmentation beyond the baseline's:** the baseline already rotates the band by −1/0/+1
   electrode (one offset per batch; upstream emg2pose draws one per window). Try per-window offsets,
   per-channel amplitude scaling, time jitter.
4. **Number of participants:** train on 25 / 50 / 100 / all train users; on Track 1, more people
   was the biggest lever for unseen people.
5. **Model size and loss:** L1 on angles in radians (the metric, and the baseline's loss) vs smooth L1; a wider or deeper network.
6. **Seed ensembles** (average of angles) for the final candidates: +2–3 points on Track 1, and steadier in the audit.

Nothing in the Track 4 baseline is pretrained (VEMG2Pose trains from scratch), so the
"fine-tune the foundation model" lesson from Track 1 doesn't apply unless we bring an EMG model.

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

## EMG2Pose subset (Oct 27)

`configs/emg2pose_subset.json` (`scripts/pick_subjects.py`, seed 0): **all 20 test users** (35 GB: local test =
the warm-up set), **all 15 val users** (26 GB: held-out selection) and **40 random train users** (71 GB).
That makes 75 participants, ~132 GB raw (0.80 GB per recorded hour), plus the extraction cache
(float32: ~170 GB, to be checked).

- Download: `make download-subset`, in your own terminal (resumable). It runs the NeuralBench CLI through
  `scripts/neuralbench_mps.py --subset`, which sets `data.study.source.query = "subject in [...]"`.
- **Every entry point needs that query.** With `query=None` the study requires all 25,253 recordings and stops
  ("Dataset ... is corrupted"), so the plain `neuralbench emg pose` CLI and benchopt's `Salter2024Emg2pose`
  don't run on a subset. **Never `benchopt prepare`**: its query is empty, so it downloads all 340 GB.
- One copy of the data: NeuralBench `DATA_DIR = data/neural_compet`, and `BENCHOPT_DATA_HOME = <repo>/data`
  (Makefile + `.venv/bin/activate`) makes benchopt's `get_data_path("neural_compet")` the same folder.
  Caches: NeuralBench's `cache/`, reused by `eval_local.py`.
- Windows in the subset (upper bounds before IK drops): train 49k (73 h), val 22k (33 h), held-out 3.7k,
  warm-up test 5.1k.

### VEMG2Pose time per epoch on `mps`

- Synthetic estimate (Oct 27): 6.0 M parameters; 0.27 s per training batch of 64 × 16 × 11,790 and 0.05 s per eval
  batch, so **~4 min/epoch GPU-bound** on the subset (771 train + 343 val batches). Reading ~110 MB per batch from
  the cache may make that longer: measure the real figure with the first run (`neuralbench_mps.py` prints
  `[epoch N] … s` lines).
- Plan: at ≤10 min/epoch, the recipe's 100 epochs (patience 20) take ≤17 h: start the day the download ends,
  upload the best smoothed-held-out epoch to the warm-up before Nov 1. If an epoch is much slower,
  cap it with `--max-neuro-hours-per-epoch` or fewer epochs and upload an earlier checkpoint.

## EMG2Pose facts (from neuralfetch / neuralbench 0.4.0)

- **Size: ~340 GB** for the 193 subjects (~1.8 GB each), BIDS `.bdf` from NEMAR, in
  `<data>/Salter2024Emg2pose/download/nm000281/`, plus a small `emg2pose_metadata.csv` (paper splits).
  A study query naming its subjects (`subject in [...]`) downloads only those.
  The preprocessed cache (`cache/`) comes on top; its size is unknown until measured.
- 25,253 recordings. Targets are the 20 joint angles stored as MISC channels in the same BDF, in radians;
  frames where inverse kinematics failed are NaN, and windows touching them are dropped.
- Splits: the paper's `split` column (train/val/test). Test keeps only `generalization == user_stage`
  (the warm-up set); val keeps both of its scenarios. Recordings shorter than 5.895 s are dropped.
- Windows: 5 s, stride 5 s; EMG scaled V → µV; no filtering or scaling.
- Baseline recipe (`neuralbench/tasks/emg/pose/config.yaml`): VEMG2Pose, Adam lr 1e-3 constant, L1 loss,
  100 epochs, patience 20 on `val/mae`, band rotation ±1. About 0.3 h/epoch on a cluster GPU, i.e. ~30 h for
  100 epochs: on a Mac, expect much longer. Measure it before planning around it.

## Open questions

- ~~Disk~~: subset of 75 participants (~132 GB + cache) on 999 GB free; more train users later (experiment 4).
- ~~One copy on disk~~: solved with `DATA_DIR = data/neural_compet` + `BENCHOPT_DATA_HOME` (see the subset section).
- Real VEMG2Pose time per epoch on `mps` (synthetic estimate: ~4 min GPU-bound), and the extraction cache size.
- ~~Pretrained?~~ No: VEMG2Pose trains from scratch.
- Who uploads what in the sealed phase (5 uploads each, separate from Track 1's).
- Whether Enrique's Mac is available for long runs, and how to sync data/checkpoints with it.
