# Results log

One row per Codabench upload. Needed to pick the best model, and for the reproducibility audit (commit + config + command must reproduce the weights).

The leaderboard ranks on **mean angular MAE in degrees** (lower is better). Compare models locally (`scripts/eval_local.py`, once it exists) before spending an upload. **Budget:** warm-up uploads are free (daily quota) but don't count; **sealed phase: 5 uploads per person in total**, team's best on the leaderboard.

| Date | Who | Phase | Commit | Model / config | Command to reproduce | Held-out MAE (°) | Local test MAE (°) | Codabench MAE (°) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27 | Margot | warm-up | 545bdbf | `submissions/meanpose_floor`: official `solvers/mean_pose.py` verbatim (competition repo `a66bc23`). No weights, so it predicts the all-zero pose | `make test-submission SUBMISSION=submissions/meanpose_floor` → `dist/meanpose_floor.zip` | n/a | not measured yet (Simulated smoke test: 20.06°) | pending | Checks Track 4 registration and the upload path only |
