"""Load a checkpoint and run :class:`~tda_ml.models.AnisotropicOutlierClassifier` forward."""

from __future__ import annotations

import logging
import os

import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.models import AnisotropicOutlierClassifier

logger = logging.getLogger(__name__)
def default_best_model_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "experiments", "best_model.pth")


def load_model(
    device: torch.device,
    weights_path: str | None = None,
) -> AnisotropicOutlierClassifier:
    """Load a trained ``AnisotropicOutlierClassifier`` (topology head outputs ``[a, b, theta]``)."""
    path = weights_path or default_best_model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pretrained weights not found: {path}\n"
            "Train and save a checkpoint, or pass an explicit weights_path."
        )

    model = AnisotropicOutlierClassifier()
    ckpt = load_torch_checkpoint(path, map_location="cpu")
    sd = extract_model_state_dict(ckpt)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s", path)
    return model


def model_forward(model: AnisotropicOutlierClassifier, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as ``model(x)``; kept for scripts that import this helper name."""
    return model(x)
