"""Ensemble of NeuralBench VEMG2Pose checkpoints: the mean of their predicted angles.

Same model, weight loading and left-context padding as the official template
(tracks/emg_pose/submission_templates/neuralbench_vemg2pose_submission.py); this loads every
``weights_*.pt`` in the submission folder and averages the members' predictions (radians).
Built by scripts/package_ensemble.py, which records the members' checkpoints in
``run_config.json`` (kept out of the ZIP).
"""

import torch
from torch.nn import functional as F
from braindecode.models import VEMG2Pose

from benchmark_utils.base_solver import CompetSolver

TRAIN_N_TIMES = 11_790
LEFT_CONTEXT = 1_790


def _load_neuralbench_weights(module, path, device):
    """Load only the brain model from a NeuralBench or plain checkpoint (as the template)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    expected = module.state_dict()
    if any(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()
                 if key.startswith("model.")}
    state = {key: value for key, value in state.items() if key in expected}
    missing = sorted(set(expected) - set(state))
    if missing:
        raise RuntimeError(f"{path.name} does not match VEMG2Pose. Missing keys: {missing}")
    module.load_state_dict(state, strict=True)


class VEMG2PoseEnsemble:
    def __init__(self, paths, n_chans, n_joints, sfreq, device):
        self.device = device
        self.nets = []
        for path in paths:
            net = VEMG2Pose(n_chans=n_chans, n_times=TRAIN_N_TIMES, n_outputs=n_joints, sfreq=sfreq).to(device)
            _load_neuralbench_weights(net, path, device)
            self.nets.append(net.eval())

    @torch.inference_mode()
    def predict(self, X):
        X = X.to(self.device, dtype=torch.float32)
        mode = "reflect" if X.shape[-1] > LEFT_CONTEXT else "replicate"
        X = F.pad(X, (LEFT_CONTEXT, 0), mode=mode)
        prediction = torch.stack([net(X) for net in self.nets]).mean(0)  # (B, T, joints), radians
        return prediction.transpose(1, 2)  # (B, joints, T), radians


class Solver(CompetSolver):

    name = "VEMG2Pose-Ensemble"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        paths = sorted(meta["submission_dir"].glob("weights_*.pt"))
        if not paths:
            raise RuntimeError("no weights_*.pt in the submission folder")
        return VEMG2PoseEnsemble(paths, meta["n_chans"], meta["n_joints"], meta["sfreq"], meta["device"])
