"""`neuralbench` CLI on a Mac: same arguments, plus a participant subset and per-epoch checkpoints.

    python scripts/neuralbench_mps.py emg pose --subset configs/emg2pose_subset.json --download
    python scripts/neuralbench_mps.py emg pose -m vemg2pose --subset configs/emg2pose_subset.json [--seeds 33]

Adapted from ../egg-to-image/scripts/neuralbench_mps.py (copied, not shared):
- Uses this repo's configs/neuralbench.local.json (``NEURALBENCH_CONFIG``, set by the
  Makefile and .venv/bin/activate; set here too if missing). Its ``CLUSTER: null`` keeps
  everything in this process, so the patches below apply. ~/.neuralbench is never read.
- ``--subset FILE`` sets ``data.study.source.query`` to FILE's ``query``
  (``subject in [...]``, from scripts/pick_subjects.py). Needed on a partial download:
  without a query the study expects all 25,253 recordings and stops ("corrupted").
  With ``--download`` only those participants are fetched (Salter2024Emg2pose reads
  the query). The query is part of the run config, so subset runs get their own cache key.
- Lightning picks ``mps`` for training on its own (accelerator="auto");
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` runs any op MPS lacks on the CPU.
- Saves the weights after every epoch (``epochs/epoch=NNN.ckpt`` in the run's folder),
  so selection can use a smoothed held-out score (scripts/eval_local.py), and prints
  each epoch's wall time.
- Copies each run's ``best.ckpt`` to ``best_kept.ckpt`` before NeuralBench deletes it.
- ``--seeds 33,34`` replaces NeuralBench's default seed grid (33, 34, 35).
"""
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO / "configs" / "neuralbench.local.json"
LOGGER = logging.getLogger(__name__)


def pop_option(name: str) -> str | None:
    """Remove ``name VALUE`` from sys.argv (the neuralbench parser doesn't know it)."""
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def use_subset(subset_file: Path) -> None:
    """Restrict every task config to the subset's participants."""
    from neuralbench import experiment_config

    query = json.loads(subset_file.read_text())["query"]
    merge = experiment_config.merge_task_config

    def merge_with_query(*args, **kwargs):
        config = merge(*args, **kwargs)
        config["data.study.source.query"] = query
        return config

    experiment_config.merge_task_config = merge_with_query
    LOGGER.warning("Study query from %s: %d subjects", subset_file, query.count("'") // 2)


def keep_best_checkpoints() -> None:
    from neuralbench.main import Experiment

    cleanup = Experiment._cleanup

    def cleanup_keeping_best(self, trainer) -> None:
        best = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if best and Path(best).exists():
            kept = Path(best).with_name("best_kept.ckpt")
            shutil.copyfile(best, kept)
            LOGGER.warning("Kept best checkpoint: %s", kept)
        cleanup(self, trainer)

    Experiment._cleanup = cleanup_keeping_best


def checkpoint_every_epoch() -> None:
    """Add a weights-only checkpoint per epoch and an epoch timer to training trainers.

    Appended after NeuralBench's own ModelCheckpoint, so ``trainer.checkpoint_callback``
    (the first one) is still its ``best.ckpt``.
    """
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from neuralbench.main import Experiment

    class EpochTimer(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            self.t0 = time.perf_counter()

        # After validation: on_validation_epoch_end runs before the module logs that epoch's
        # val metrics, so it would print the previous epoch's.
        def on_train_epoch_end(self, trainer, module):
            if not hasattr(self, "t0"):
                return
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
            print(f"[epoch {trainer.current_epoch}] {time.perf_counter() - self.t0:.0f} s (train + val), "
                  + ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())), flush=True)

    setup = Experiment.setup_trainer

    def setup_with_epoch_checkpoints(self, is_test=False, limit_train_batches=None):
        trainer = setup(self, is_test=is_test, limit_train_batches=limit_train_batches)
        if not is_test and self.infra.uid_folder() is not None:
            trainer.callbacks.append(ModelCheckpoint(
                dirpath=Path(self.infra.uid_folder()) / "epochs", filename="{epoch:03d}",
                save_top_k=-1, every_n_epochs=1, save_weights_only=True,
                enable_version_counter=False))
            trainer.callbacks.append(EpochTimer())
        return trainer

    Experiment.setup_trainer = setup_with_epoch_checkpoints


def override_seed_grid(seeds: list[int]) -> None:
    """Replace NeuralBench's default seed grid (33, 34, 35) with ``seeds``."""
    from neuralbench import experiment_config

    load = experiment_config.load_yaml_config

    def load_with_seeds(path, *args, **kwargs):
        config = load(path, *args, **kwargs)
        if Path(path) == experiment_config.DEFAULTS_DIR / "grid.yaml":
            config["seed"] = seeds
        return config

    experiment_config.load_yaml_config = load_with_seeds


if __name__ == "__main__":
    os.environ.setdefault("NEURALBENCH_CONFIG", str(LOCAL_CONFIG))
    os.environ.setdefault("BENCHOPT_DATA_HOME", str(REPO / "data"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if not Path(os.environ["NEURALBENCH_CONFIG"]).exists():
        sys.exit(f"{os.environ['NEURALBENCH_CONFIG']} is missing: run `make neuralbench-config`")

    subset = pop_option("--subset")
    seeds = pop_option("--seeds")
    if subset:
        use_subset(Path(subset))
    if seeds:
        override_seed_grid([int(s) for s in seeds.split(",")])
    keep_best_checkpoints()
    checkpoint_every_epoch()

    from neuralbench import run_benchmark_cli

    sys.argv[0] = "neuralbench"
    run_benchmark_cli()
