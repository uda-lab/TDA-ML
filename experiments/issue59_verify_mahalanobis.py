#!/usr/bin/env python3
"""
Issue #59 supervised verification (mahalanobis, reproduce_1week_tuned hyperparameters).

Runs training with early abort when val/train MCC stay near zero after probe_epochs.
On abort, writes ``logs/issue59_abort_report.{json,md}`` with ellipse / encoder /
distance / classification diagnostics and ranked failure hypotheses.

Usage::

    uv run python experiments/issue59_verify_mahalanobis.py
    uv run python experiments/issue59_verify_mahalanobis.py --epochs 12 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tda_ml.config import deep_update, load_config
from tda_ml.main import main as train_main
from tda_ml.supervised_diagnostics import git_revision

REPO = Path(__file__).resolve().parents[1]


def preflight(config_name: str) -> dict:
    cfg = load_config(config_name, project_root=REPO)
    data = cfg.get("data", {})
    training = cfg.get("training", {})
    early = training.get("early_abort", {})
    if not early.get("enabled", False):
        raise ValueError(
            f"{config_name}: training.early_abort.enabled must be true for verification"
        )
    backend = (
        cfg.get("model", {}).get("topology_loss", {}).get("distance_backend", "")
    )
    if backend != "mahalanobis":
        raise ValueError(
            f"issue59 verification expects mahalanobis backend; got {backend!r}"
        )
    data_root = REPO / "data"
    if not data_root.exists():
        raise FileNotFoundError(
            f"MNIST data root missing: {data_root}. Run training once to download."
        )
    return {
        "config_id": cfg["meta"].get("config_id"),
        "source_revision": git_revision(REPO),
        "epochs": training.get("epochs"),
        "seed": data.get("seed"),
        "early_abort": early,
        "distance_backend": backend,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="issue59_verify_mahalanobis",
        help="Config name under configs/ (default: issue59_verify_mahalanobis)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=REPO / "outputs" / "issue59_verify",
    )
    args = parser.parse_args()

    manifest_preview = preflight(args.config)
    print("=== Preflight OK ===")
    print(json.dumps(manifest_preview, indent=2, ensure_ascii=True))

    cfg = load_config(args.config, project_root=REPO)
    overrides: dict = {"outputs": {"base_dir": str(args.out_base)}}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.seed is not None:
        overrides.setdefault("data", {})["seed"] = args.seed
    cfg = deep_update(cfg, overrides)

    result = train_main(config=cfg)
    status = result.get("status", "completed")
    print("\n=== Verification result ===")
    print(f"status: {status}")
    print(f"run_dir: {result.get('run_dir')}")
    print(f"best_val_mcc: {result.get('best_val_mcc', result.get('val_mcc')):.4f}")
    if result.get("abort_report"):
        print(f"abort_report: {result['abort_report']}")
        print("\nReview hypotheses in issue59_abort_report.md before long runs.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
