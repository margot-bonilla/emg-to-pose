# Results log

One row per Codabench upload. Needed to pick the best model, and for the reproducibility audit (commit + config + command must reproduce the weights).

The leaderboard ranks on **mean angular MAE in degrees** (lower is better). Compare models locally (`scripts/eval_local.py`, once it exists) before spending an upload. **Budget:** warm-up uploads are free (daily quota) but don't count; **sealed phase: 5 uploads per person in total**, team's best on the leaderboard.

| Date | Who | Phase | Commit | Model / config | Command to reproduce | Held-out MAE (°) | Local test MAE (°) | Codabench MAE (°) | Notes |
|---|---|---|---|---|---|---|---|---|---|
