"""Write this repo's own NeuralBench config, configs/neuralbench.local.json.

NeuralBench reads the file named by the NEURALBENCH_CONFIG environment variable
before falling back to the global ~/.neuralbench/config.json. The Makefile and
.venv/bin/activate export NEURALBENCH_CONFIG pointing here, so this repo keeps
its own data/, cache/ and save/ and never depends on another project's config
(e.g. Track 1 on the same Mac). ~/.neuralbench is never read or written.

DATA_DIR is data/neural_compet: benchopt's Salter2024Emg2pose reads
get_data_path("neural_compet") = $BENCHOPT_DATA_HOME/neural_compet, and the Makefile
and .venv/bin/activate export BENCHOPT_DATA_HOME=<repo>/data, so NeuralBench, benchopt
and scripts/eval_local.py share one copy of the data.

CLUSTER is null: everything runs locally, in-process (no Slurm, and no exca
subprocess where patches such as the Mac GPU one would not apply).

An existing file is left untouched unless --force is given.

    python scripts/neuralbench_config.py [--force]
"""
import argparse
import getpass
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "neuralbench.local.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing config")
    args = parser.parse_args()

    if CONFIG.exists() and not args.force:
        print(f"{CONFIG} already exists, leaving it as is (use --force to overwrite)")
        raise SystemExit(0)

    user = getpass.getuser()
    config = {
        "USER": user,
        "ENTITY_NAME": user,
        "PROJECT_NAME": "neuralbench",
        "DATA_DIR": str(REPO / "data" / "neural_compet"),
        "CACHE_DIR": str(REPO / "cache"),
        "SAVE_DIR": str(REPO / "save"),
        "WANDB_HOST": "",
        "SLURM_PARTITION": "",
        "SLURM_CONSTRAINT": "",
        "N_CPUS": os.cpu_count(),
        "CLUSTER": None,
    }
    for key in ("DATA_DIR", "CACHE_DIR", "SAVE_DIR"):
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {CONFIG}")
