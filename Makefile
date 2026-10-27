PYTHON := .venv/bin/python
PIP := .venv/bin/pip

# This repo's own NeuralBench config (data/, cache/, save/ here), independent of
# ~/.neuralbench and of any other project: see scripts/neuralbench_config.py.
export NEURALBENCH_CONFIG := $(CURDIR)/configs/neuralbench.local.json
# benchopt's Salter2024Emg2pose reads $(BENCHOPT_DATA_HOME)/neural_compet, NeuralBench's
# DATA_DIR: one copy of the data for both (never run `benchopt prepare` on it, see PLAN.md).
export BENCHOPT_DATA_HOME := $(CURDIR)/data
# The fbcode Python 3.12 looks for CA certificates in /etc/pki/tls (Linux), absent on macOS:
# urllib then fails with CERTIFICATE_VERIFY_FAILED (e.g. EMG2Pose's BIDS sidecars from NEMAR).
# certifi's bundle fixes it; an SSL_CERT_FILE already set wins.
export SSL_CERT_FILE ?= $(shell $(PYTHON) -m certifi 2>/dev/null)

.PHONY: env env-create env-install neuralbench-config competition-repo starter download-subset zip-submission test-submission pause-training resume-training training-status clean help

help:
	@echo "Available commands:"
	@echo "  make env              - Python 3.12 venv, dependencies, competition repo, NeuralBench config"
	@echo "  make starter          - Zero-download contract check (official starter.yml: MeanPose on Simulated)"
	@echo "  make download-subset  - EMG2Pose participants of configs/emg2pose_subset.json (~132 GB; run in your own terminal)"
	@echo "  make zip-submission   - Package a submission folder for Codabench upload"
	@echo "  make test-submission  - Zip, then run it read-only like Codabench does"
	@echo "  make pause-training   - Freeze all training runs on this Mac, Track 1's too (frees the GPU; nothing is lost)"
	@echo "  make resume-training  - Continue them where they stopped"
	@echo "  make training-status  - All runs on this Mac: project, running / paused, latest epoch. Check before any GPU job"
	@echo "  make clean            - Remove temporary caches"

env: env-create env-install competition-repo neuralbench-config

# Python 3.12 matches the Codabench worker image (python:3.12-slim); newer
# Pythons may lack wheels for the torch / neuroai stack. Override with
# make env-create PY312=/path/to/python3.12
PY312 ?= python3.12

env-create:
	@command -v $(PY312) > /dev/null || { echo "Need Python 3.12 on PATH (e.g. brew install python@3.12, or make env-create PY312=...)"; exit 1; }
	@$(PY312) -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' || \
		{ echo "Need Python 3.12, found $$($(PY312) --version 2>&1)"; exit 1; }
	@test -d .venv || { echo "$(PY312) -m venv .venv"; $(PY312) -m venv .venv; }

env-install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Writes configs/neuralbench.local.json (never overwrites) and makes
# `source .venv/bin/activate` export NEURALBENCH_CONFIG, so the neuralbench CLI
# in a terminal uses this repo's config too.
neuralbench-config:
	$(PYTHON) scripts/neuralbench_config.py
	@grep -q NEURALBENCH_CONFIG .venv/bin/activate || \
		echo 'export NEURALBENCH_CONFIG="$(NEURALBENCH_CONFIG)"' >> .venv/bin/activate
	@grep -q BENCHOPT_DATA_HOME .venv/bin/activate || \
		echo 'export BENCHOPT_DATA_HOME="$(BENCHOPT_DATA_HOME)"' >> .venv/bin/activate
	@grep -q SSL_CERT_FILE .venv/bin/activate || \
		echo 'export SSL_CERT_FILE="$${SSL_CERT_FILE:-$$(python -m certifi)}"' >> .venv/bin/activate

# Official competition repo: provides benchmark_utils, the scoring objective and
# the datasets used by Codabench. Kept out of git (see .gitignore).
COMPET_REPO := external/2026-competition
TRACK := tracks/emg_pose

competition-repo:
	@test -d $(COMPET_REPO) || git clone --depth 1 https://github.com/neural-interfaces26/2026-competition $(COMPET_REPO)

starter: competition-repo
	cd $(COMPET_REPO) && PATH="$(CURDIR)/.venv/bin:$$PATH" \
		benchopt run $(TRACK) --config $(TRACK)/starter.yml --no-plot --no-cache

# Only the participants listed in configs/emg2pose_subset.json (scripts/pick_subjects.py):
# the NeuralBench CLI with their `subject in [...]` query. Idempotent: re-run to resume.
# neuralfetch skips the whole download once its per-study success marker exists, even for
# subjects added since: remove it first (eegdash still skips complete recordings).
SUBSET ?= configs/emg2pose_subset.json
SUCCESS_MARKER := data/neural_compet/Salter2024Emg2pose/download/eegdash_nm000281_success_download.txt

download-subset:
	rm -f $(SUCCESS_MARKER)
	$(PYTHON) scripts/neuralbench_mps.py emg pose --subset $(SUBSET) --download
	$(PYTHON) scripts/check_subset.py --subset $(SUBSET)

# Folder holding submission.py + its weights/artifacts, e.g. make test-submission SUBMISSION=submissions/<name>
# Every file in it goes into the ZIP except run_config* (kept for the audit) and dotfiles.
SUBMISSION ?= submission
# Synthetic data with the real input shape: 16 channels x 10,000 samples (5 s at 2 kHz) -> 20 joints.
SIM := Simulated['n_chans, n_joints, n_times'=[(16, 20, 10000)], 'n_train, n_test'=[(20, 40)], sfreq=2000.0]

zip-submission:
	@mkdir -p dist
	rm -f dist/submission.zip
	zip -j dist/submission.zip $$(find $(SUBMISSION) -maxdepth 1 -type f ! -name 'run_config*' ! -name '.*')
	@echo "Created dist/submission.zip from $(SUBMISSION)/, ready for Codabench upload!"

# Mimic Codabench: unzip into a read-only folder and run inference-only.
test-submission: competition-repo zip-submission
	rm -rf dist/unzipped && mkdir -p dist/unzipped
	unzip -q dist/submission.zip -d dist/unzipped
	chmod -R a-w dist/unzipped
	cd $(COMPET_REPO) && PATH="$(CURDIR)/.venv/bin:$$PATH" \
		COMPET_SUBMISSION_DIR="$(CURDIR)/dist/unzipped" \
		benchopt run $(TRACK) -d "$(SIM)" \
		-s "$(CURDIR)/dist/unzipped/submission.py" --no-plot --no-cache
	chmod -R u+w dist/unzipped

# Local training runs (scripts/train_solver.py), machine-wide: see scripts/training.sh.
# Track 1 keeps GPU priority: never start a GPU job here while it trains.
pause-training:
	@scripts/training.sh pause

resume-training:
	@scripts/training.sh resume

training-status:
	@scripts/training.sh status

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	chmod -R u+w dist 2>/dev/null || true
	rm -rf dist .pytest_cache
