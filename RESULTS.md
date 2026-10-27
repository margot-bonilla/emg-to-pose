# Results log

One row per Codabench upload. Needed to pick the best model, and for the reproducibility audit (commit + config + command must reproduce the weights).

The Codabench Score is the **mean over subjects of the angular MAE in degrees** (`angular_mae_subject_mean`, lower is better). Compare models locally (`scripts/eval_local.py`, once it exists) before spending an upload. **Budget:** warm-up uploads are free (daily quota) but don't count; **sealed phase: 5 uploads per person in total**, team's best on the leaderboard.

| Date | Who | Phase | Commit | Model / config | Command to reproduce | Held-out MAE (°) | Local test MAE (°) | Codabench MAE (°) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27 | Margot | warm-up | 545bdbf | `submissions/meanpose_floor`: official `solvers/mean_pose.py` verbatim (competition repo `a66bc23`). No weights, so it predicts the all-zero pose | `make test-submission SUBMISSION=submissions/meanpose_floor` → `dist/meanpose_floor.zip` | n/a | **22.6283** (per-subject mean; pooled 22.6666) | **22.63** (Codabench #1012775) | Registration + upload path OK. Local = Codabench (after the BDF unit fix, see PLAN) |
