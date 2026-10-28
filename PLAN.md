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

## Road to the sealed phase (plan of Oct 28)

**Goal:** the best score on the sealed 2026 cohort (new users, new stages incl. some outside the 29 public ones, possibly a
different recording setup) *and* passing the audit (retrain from committed code + config within ±2σ). The warm-up only
checks packaging (local = Codabench on 4 uploads).

**Principles**
1. **Two proxies for the sealed shift, not one:** held-out (val users × held-out stages) plus a **stressed held-out**: the same
   windows with simulated recording shifts (electrode rotation ±1–2, per-channel gain ×0.5–2, noise, small time shifts).
   A recipe must win on both; clean held-out alone rewards fitting EMG2Pose's own setup.
2. **What we measured works, keep it:** more users (40 → 100 → 158: 18.4 → 17.3 → 16.8 held-out) and ensembles (one run's
   snapshots −0.5°, mixing runs −0.4° more). Models peak early on unseen people (plateau from ~epoch 11–18).
3. **Sealed uploads are our only signal on the real cohort:** 15 uploads (5 × 3 people). Spend them as controlled
   comparisons (one factor at a time), keep spares for the end, never upload an untested ZIP (`make test-submission`).
4. **Audit-ready by construction:** one command per ensemble member, fixed seeds and epoch counts, `run_config.json` per
   submission, nothing Mac-only (must retrain on CUDA too), the NEMAR unit fix in the data pipeline.

**Phases**
| When | What | Done when |
|---|---|---|
| Oct 28–29 | `scripts/train_vemg2pose.py`: own loop on NeuralBench's cached windows (same data, same VEMG2Pose): per-epoch checkpoints + resume, cosine LR, EMA weights, augmentations (per-window band rotation, per-channel gain, noise, time shift), optional per-window normalisation (also in `submission.py`), all-data mode (all users × all stages, fixed epochs). Plus the stressed held-out in `eval_local.py` | a short run reproduces the NeuralBench baseline on held-out within noise; current ensembles scored on the stressed held-out |
| — | **Stressed held-out, first result (Oct 28, single 158-user model, epoch 25, pooled):** clean 16.68; rotation ±1 16.82–16.94; **per-channel gain 0.5–2× 17.58 (+0.9)**; global gain ×0.5 17.14, ×2 16.71; noise 20% 16.93; mean 17.02. Amplitude is the weak spot: E1 (gain aug) and E2 (per-window norm) target it | — |
| Oct 29–31 | Experiments at 158 users, ~15 epochs (~2.5 h) each, judged on held-out + stressed held-out: (E1) augmentation, (E2) per-window normalisation, (E3) cosine LR + EMA, (E4) test-time adaptation per batch (eval-only, on E2 models) | a recipe chosen on Oct 31 |
| Nov 1 | Last warm-up upload: the chosen recipe's packaging (local = Codabench) | match |
| Nov 1–4 | Final training on **all 193 users × all 29 stages**, 3–5 seeds × fixed epochs (from the curves), ensembles of seeds × snapshots. ~6 h per seed: needs a second Mac or nights | 2–3 candidates: best recipe, a diverse variant, the safe current recipe on all data; audit commands written |
| Nov 5–21 | Sealed uploads: day 1 best + safe (tells us how held-out transfers to the 2026 cohort), then single-factor comparisons (normalisation, TTA, shrinkage α), keep 3–4 for the last week | best on the leaderboard, audit pack ready |

## Rules and lessons (moved from the Track 1 handoff doc, Oct 27)

**Competition:** sealed uploads are 5 per person per track **in total**, and failed uploads still use a slot:
always `make test-submission` first (ZIP = `submission.py` + weights at its root, `run_config*` stays out).
Top 3 get a reproducibility audit: the organisers retrain from the committed code + config and must land within
±2σ, so history is never rewritten (no squash, rebase or force-push: RESULTS.md and `run_config.json` cite exact commits). Codabench
had a multi-day outage around Oct 22: check that an upload was actually scored. Official Track 4 pages:
`codabench/pages/competition_emg.md`, `codabench/pages/scientific_faq_emg.md` in the competition repo.

**Track 1 lessons still to apply here:**
- More participants was the biggest lever for unseen people (+3.6 points per doubling): measure the curve.
- Per-window normalisation helped unseen people a lot on one dataset (+4 points), not on another: test it early.
- Models peak much earlier on unseen people; a final retrain without validation needs a fixed epoch count taken
  from the smoothed held-out curves. Seed ensembles added 2–3 points and survive the audit better.
- Foundation models: fine-tune, don't just probe; see `neuralbench/defaults/downstream_wrappers.yaml`.
- Cache windows before training (~4× faster on Track 1; here NeuralBench's cache already keeps up at 195 s/epoch).
- The lid closed puts macOS to sleep despite `caffeinate`: keep it open for overnight runs. The screen blinking
  means the GPU is at ~97%: `make pause-training`, don't restart.
- Never edit a running bash queue script (bash reads it as it goes); replace imported Python modules atomically.

**From the official pages (Oct 27):** label-free test-time adaptation inside the current batch is allowed, gradient updates
included (no labels, no reliance on row order or state across calls); inference must finish in 60 min on one A100; 15 GB
submission storage per profile; public external EMG data (e.g. NinaPro) allowed if declared; the sealed cohort has stages
outside the 29 public ones. Each participant has their own upload quota, ranked as the team.

## Constraints

- **GPU:** Track 1 has priority on Margot's Mac, especially at night. Run `make training-status`
  (lists every run on the Mac) before any GPU job; never run two trainings at once.
- **No dependency on Track 1:** own config, data and caches (duplicated if needed); files are copied, never linked.
- **Workflow:** push directly to `main`, no PRs (decided Oct 27: two of us, no reviewers); never rewrite history.
  Heavy Track 4 training belongs on Enrique's Mac or in daytime gaps.
- **Downloads:** Margot runs them in her own terminal.
- **Disk:** 999 GB free (Oct 27), shared with Track 1's caches.

## Milestones

| # | When | Goal | Done when |
|---|---|---|---|
| 1 ✅ | Oct 22 | `make env`, official repo, `make starter` | `starter.yml` (MeanPose on Simulated) passes: 20.8° on Simulated |
| 2 | Oct 27–28 | EMG2Pose subset (75 participants, ~132 GB, see below): `make download-subset`, then `--prepare` | the `emg/pose` task loads a train and a test batch |
| 3 ✅ | Oct 27 | `scripts/eval_local.py` + NeuralBench VEMG2Pose on the subset (`scripts/neuralbench_mps.py`, `scripts/package_vemg2pose.py` ✅) | one warm-up upload where local MAE = Codabench MAE; then delete `docs/handoff_track4_emg_pose.md` (its useful parts live in README/PLAN) | **Done Oct 27:** floor 22.6283 local / 22.63 Codabench; VEMG2Pose epoch 1 18.5202 / 18.52 |
| 4 ✅ | Oct 27 | Held-out split = the paper's val users × held-out stages (`eval_local.py --split heldout`) | every run reports a held-out MAE |
| 5 | Oct 26–28 | Window cache, `scripts/train_solver.py`, `submission/submission.py` with per-epoch checkpoints, `--resume` and smoothed early stopping | a restartable run that exports an upload-ready folder with `run_config.json` |
| 6 | Oct 28 – Nov 1 | Experiments on held-out (list below) | a recipe that clearly beats the baseline on held-out |
| 7 | Nov 2–4 | Final retrains on all users (fixed epoch count), seed ensembles, `make test-submission` | 2–3 candidates in `submissions/`, logged in `RESULTS.md` |

Sealed phase: upload 2–3 very different candidates early, refine what wins, end with the best.

## Warm-up uploads

- Oct 27: `dist/meanpose_floor.zip` (official MeanPose verbatim; with no weights it predicts the all-zero pose): **22.63°** on Codabench (#1012775). Registration and upload path OK; `eval_local.py --split test` must reproduce 22.63.
- Oct 27: `dist/vemg2pose_s33_ep001.zip` (VEMG2Pose, subset, seed 33, epoch 1): **18.52°** on Codabench (#1013210) = local 18.5202.
- Oct 27: `dist/vemg2pose_u100_s33_ep012.zip` (VEMG2Pose, 100 train users, seed 33, epoch 12): **17.37°** on Codabench (#1013852) = local 17.3681.
- Oct 28: `dist/vemg2pose_u100_s33_snap5.zip` (5-epoch snapshot ensemble, 100 users): **16.66°** on Codabench (#1015483) = local 16.6584 per-subject (16.54 pooled).

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

**Where we stand (Oct 27, VEMG2Pose seed 33, epochs 0–9, held-out = val users × held-out stages, per-subject MAE):**
constant train-mean pose 19.50° (test 19.16°); VEMG2Pose 18.4–20.0° per epoch, best smoothed 18.79° (epoch 5), and
R² < 0 throughout, while NeuralBench's pooled val (mostly seen stages) keeps improving. So the model barely transfers to
unseen stages: everything below should be judged on held-out only. Cheap finishing touch: shrink predictions toward the
train mean, `mean + α (pred − mean)`; α = 0.75 gains 0.1–0.2° on held-out (epochs 4, 5, 9). Choose α on held-out.

**Seed noise (seed 34, same 40-user recipe, epochs 0–15):** best smoothed held-out 18.44° (epoch 9) vs seed 33's 18.64°;
single epochs swing 18.3–20.0°. So a change must move the *smoothed* held-out curve clearly below ~18.4° to count.
Ensemble of the two seeds' best epochs (s33 e10 + s34 e9): **18.20°** vs 18.52 / 18.40 alone (−0.2 to −0.3°).

**Participants (experiment 4), launched Oct 27 15:38:** `configs/emg2pose_subset_100.json` = same test/val users + 100 train
users (the 40 + 60 new, `pick_subjects.py --keep`), 17,908 recordings (11,075 train) verified by `check_subset.py`.
Run: `scripts/detach.py logs/neuralbench_vemg2pose_u100_s33.log -- .venv/bin/python scripts/neuralbench_mps.py emg pose
-m vemg2pose --subset configs/emg2pose_subset_100.json --seeds 33`. Compare its smoothed held-out curve with s33/s34 (40 users).
**Result (epochs 0–15):** best smoothed held-out **17.80°** (epoch 12) vs 18.44 / 18.64 with 40 users, every epoch since 6
below the 40-user best; R² −0.02 to −0.05. More users is the strongest lever so far (as on Track 1): next, all 158
train users (~+105 GB). Candidate `submissions/vemg2pose_u100_s33_ep012`: local warm-up **17.37°** (NeuroPose ref. 17.5).

**100-user run, all 50 epochs (Oct 28):** held-out flat from epoch ~11 (pooled smoothed 17.30–17.75; best epoch 31, 17.30,
within noise of epoch 12's 17.38): the model peaks early on unseen people. **Snapshot ensemble** of 5 epochs of that one run
(12, 23, 31, 44, 49): **16.80 pooled** held-out vs 17.30–17.46 alone, **−0.5° for free**; local warm-up 16.54 pooled /
16.66 per-subject (`submissions/vemg2pose_u100_s33_snap5`, `scripts/package_ensemble.py`). Keep every epoch checkpoint.

**158-user run (Oct 28, epochs 0–19):** pooled held-out smoothed **16.99** at epoch 19 and still dropping (100 users: 17.49
at epoch 19, best 17.30 over 50). Ensembles on held-out (pooled): 5 checkpoints of the 158-user run (11, 14, 17, 18, 19)
16.66; those + the 100-user 5 (10 members) **16.54** (mixing runs beats one run). Next candidate once the 158-user run
plateaus (~epoch 30): best plateau checkpoints of both runs.
**Stopped at epoch 30 (Oct 28 15:30):** held-out flat 16.8–17.1 from epoch 18, best smoothed 16.83 (epoch 25). Candidate
`submissions/vemg2pose_mix_u158u100_10` (158-user epochs 18, 23, 24, 25, 28 + the 100-user 5): held-out **16.40** pooled
(12 members 16.40, 158-only 7 members 16.44); local warm-up 16.18 pooled / 16.30 per-subject.

**Next (Oct 27): all 158 train users** (`configs/emg2pose_subset_158.json` = all 193 participants, ~339 GB raw):
`make download-subset SUBSET=configs/emg2pose_subset_158.json`, then the same run with that subset.
**Final retrain (Nov 2–4), not measurable on held-out:** all 193 users × all 29 stages (add the train users' held-out-stage
recordings, which the task drops, and the val/test users), fixed epoch count from the smoothed held-out curves.


1. ~~**Train on 5 s windows without left context.**~~ Measured Oct 27 (`scripts/diag_left_context.py`, epoch 7, held-out): real context 19.32° vs the template's reflection padding 19.07°, so no loss; low priority. The gap that matters is unseen stages: ~15° on NeuralBench's pooled val (mostly seen stages) vs ~19° held-out.
   Original idea: NeuralBench's VEMG2Pose trains on 5.895 s windows
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
- Download done Oct 27 (126 GB, ~1 h at ~35 MB/s); `scripts/check_subset.py`: 9,819 recordings for 75 participants = the metadata (test user_stage 456, val user_stage 332, train 4,335). The query must name subjects `Salter2024Emg2pose/<label>`.
- Windows in the subset (upper bounds before IK drops): train 49k (73 h), val 22k (33 h), held-out 3.7k,
  warm-up test 5.1k.

### NEMAR joint-angle units (found Oct 27, fixed on disk)

Every nm000281 BDF stores the 20 joint angles in **micro-radians**. Old files label them `uV` (MNE divides by
1e6, so radians: what NeuralBench, VEMG2Pose and Codabench use); files re-uploaded since label them `rad`
without rescaling, so they read **1e6 too large**. That was 94% of our download (mixed even within a subject:
NEMAR is mid re-upload). `scripts/fix_bdf_joint_units.py` rewrites those headers' unit field to `uV` (samples
untouched, 9,373 files, every change in `download/nm000281/joint_unit_fix.csv`, idempotent). Run it after
**every** download, then delete any target cache built before. 142 files with a 0..1 placeholder range
(all-zero, fully `BAD_IK`) are left alone: their windows are dropped. Worth reporting to NEMAR / the organisers.

After the fix, `eval_local.py submissions/meanpose_floor --split test` = **22.6283** per-subject mean
(Codabench 22.63). The My Submissions "Score" is the per-subject mean `angular_mae_subject_mean`; the **official ranking
metric is the pooled `angular_mae`** (codabench/pages/competition_emg.md): select on the pooled value (`eval_local.py` does, since Oct 27).

### VEMG2Pose time per epoch on `mps`

- Synthetic estimate (Oct 27): 6.0 M parameters; 0.27 s per training batch of 64 × 16 × 11,790 and 0.05 s per eval
  batch, so **~4 min/epoch GPU-bound** on the subset (771 train + 343 val batches). Reading ~110 MB per batch from
  the cache may make that longer: measure the real figure with the first run (`neuralbench_mps.py` prints
  `[epoch N] … s` lines).
- **Measured (Oct 27, run seed 33): 195 s/epoch** (train + val) on `mps`: 23,304 train windows (5.895 s, after IK
  drops), 9,643 val; data loading keeps up. Cache: ~76 GB EMG + ~95 GB targets, built in ~35 min.
  First run: `scripts/neuralbench_mps.py emg pose -m vemg2pose --subset configs/emg2pose_subset.json --seeds 33`,
  started 13:06, launched in its own session (`start_new_session`): runs tied to Claude's task shell die with it.
  epoch 0: val/mae 0.283 rad (16.2°, pooled val); 100 epochs ≤ 5.5 h.
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
