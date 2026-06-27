#!/usr/bin/env python3
"""
Paper-aligned evaluation: ellphi DBSCAN inference + MCC / G-Mean / W-Dist.

Trainer ``val_mcc`` uses sigmoid(logit) > threshold; the paper reports metrics
after DBSCAN on the learned ellphi precomputed distance matrix.

Usage::

    uv run python experiments/evaluate_paper_protocol.py \\
        --run-dir outputs/paper_reproduce_1week_tuned/backend_ellphi_seed42_* \\
        --base-config reproduce \\
        --split val

    uv run python experiments/evaluate_paper_protocol.py \\
        --run-dir ... --split test \\
        --dbscan-hparams outputs/.../logs/dbscan_hparams.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import deep_update, load_config, model_kwargs_from_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.dbscan import apply_anisotropic_dbscan
from tda_ml.metrics import compute_recall_specificity_gmean_mcc_wdist
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.seed_utils import set_global_seed
from tda_ml.supervised_diagnostics import git_revision

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CloudMetrics:
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float


@dataclass
class SplitMetrics:
    split: str
    n_clouds: int
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float
    dbscan_eps: float | None = None
    dbscan_min_samples: int | None = None
    backend: str = "ellphi"


def _valid_clean_inliers(clean_pc: np.ndarray) -> np.ndarray:
    mask = np.abs(clean_pc).sum(axis=1) > 1e-6
    return clean_pc[mask]


def dbscan_labels_to_outlier_pred(labels: np.ndarray) -> np.ndarray:
    """DBSCAN noise (-1) -> outlier (1); clustered points -> inlier (0)."""
    return (labels == -1).astype(np.int64)


def evaluate_cloud_dbscan(
    points: np.ndarray,
    params: np.ndarray,
    labels_gt: np.ndarray,
    clean_pc: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    backend: str = "ellphi",
    metric: str = "max",
) -> CloudMetrics:
    db_labels = apply_anisotropic_dbscan(
        points,
        params,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        backend=backend,
    )
    pred = dbscan_labels_to_outlier_pred(db_labels)
    gt_inliers = _valid_clean_inliers(clean_pc)
    recall, specificity, gmean, mcc, wdist = compute_recall_specificity_gmean_mcc_wdist(
        labels_gt,
        pred,
        points=points,
        gt_inliers=gt_inliers,
    )
    return CloudMetrics(recall, specificity, gmean, mcc, wdist)


def _aggregate_cloud_metrics(rows: list[CloudMetrics]) -> tuple[float, float, float, float, float]:
    if not rows:
        raise ValueError("No clouds to aggregate")
    return (
        float(np.mean([r.recall for r in rows])),
        float(np.mean([r.specificity for r in rows])),
        float(np.mean([r.gmean for r in rows])),
        float(np.mean([r.mcc for r in rows])),
        float(np.mean([r.wdist for r in rows])),
    )


def build_split_loader(config: dict[str, Any], split: str, device: torch.device):
    data_cfg = config["data"]
    seed = int(data_cfg.get("seed", 42))
    set_global_seed(seed, deterministic_algorithms=False)

    train_size = int(data_cfg.get("train_size", 4500))
    val_size = int(data_cfg.get("val_size", 500))
    test_size = int(data_cfg.get("test_size", 1000))
    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[: train_size + val_size]
    train_indices = full_train_indices[:train_size]
    val_indices = full_train_indices[train_size:]
    test_indices = torch.randperm(10000, generator=generator)[:test_size]

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    batch_size = int(data_cfg.get("batch_size", 64))

    if split == "val":
        indices = val_indices
        train_flag = True
    elif split == "test":
        indices = test_indices
        train_flag = False
    else:
        raise ValueError(f"split must be 'val' or 'test'; got {split!r}")

    dataset = NoisyMNISTDataset(
        root=str(REPO_ROOT / "data"),
        train=train_flag,
        max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"],
        indices=indices,
        deterministic=True,
        noise_seed=seed,
        preload=True,
    )
    loader = create_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader


def load_model_from_run(run_dir: Path, config: dict[str, Any], device: torch.device) -> AnisotropicOutlierClassifier:
    ckpt_path = run_dir / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = AnisotropicOutlierClassifier(**model_kwargs_from_config(config))
    ckpt = load_torch_checkpoint(str(ckpt_path), map_location="cpu")
    model.load_state_dict(extract_model_state_dict(ckpt), strict=False)
    model.to(device)
    model.eval()
    return model


def iter_cloud_predictions(
    model: AnisotropicOutlierClassifier,
    loader,
    device: torch.device,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    with torch.no_grad():
        for data, labels, clean_pc in loader:
            data = data.to(device, non_blocking=True)
            _, params = model(data)
            data_np = data.cpu().numpy()
            params_np = params.cpu().numpy()
            labels_np = labels.cpu().numpy()
            clean_np = clean_pc.cpu().numpy()
            batch_size = data_np.shape[0]
            for b in range(batch_size):
                yield (
                    data_np[b],
                    params_np[b],
                    labels_np[b],
                    clean_np[b],
                )


def grid_search_dbscan(
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    eps_values: list[float],
    min_samples_values: list[int],
    backend: str = "ellphi",
) -> tuple[float, int, float]:
    best_mcc = -1.0
    best_eps = eps_values[0]
    best_min_samples = min_samples_values[0]
    grid_log: list[dict[str, Any]] = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            per_cloud: list[CloudMetrics] = []
            for points, params, labels_gt, clean_pc in clouds:
                try:
                    m = evaluate_cloud_dbscan(
                        points,
                        params,
                        labels_gt,
                        clean_pc,
                        eps=eps,
                        min_samples=min_samples,
                        backend=backend,
                    )
                    per_cloud.append(m)
                except Exception as exc:  # noqa: BLE001 — log and skip bad hparams
                    grid_log.append(
                        {
                            "eps": eps,
                            "min_samples": min_samples,
                            "error": str(exc),
                        }
                    )
                    per_cloud = []
                    break
            if not per_cloud:
                continue
            _, _, _, mcc, _ = _aggregate_cloud_metrics(per_cloud)
            grid_log.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "mean_mcc": mcc,
                    "n_clouds": len(per_cloud),
                }
            )
            if mcc > best_mcc:
                best_mcc = mcc
                best_eps = eps
                best_min_samples = min_samples

    if best_mcc < 0:
        raise RuntimeError(f"DBSCAN grid search failed for all hparams; log={grid_log[:5]}")
    return best_eps, best_min_samples, best_mcc


def evaluate_split(
    run_dir: Path,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    *,
    eps: float | None = None,
    min_samples: int | None = None,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    backend: str = "ellphi",
) -> SplitMetrics:
    loader = build_split_loader(config, split, device)
    model = load_model_from_run(run_dir, config, device)
    clouds = list(iter_cloud_predictions(model, loader, device))

    if split == "val":
        eps_values = eps_values or list(np.linspace(0.15, 1.5, 15))
        min_samples_values = min_samples_values or [3, 5, 7, 10, 15]
        eps, min_samples, _ = grid_search_dbscan(
            clouds,
            eps_values=eps_values,
            min_samples_values=min_samples_values,
            backend=backend,
        )
    else:
        if eps is None or min_samples is None:
            raise ValueError("test split requires eps and min_samples (from val tuning)")

    per_cloud: list[CloudMetrics] = []
    for points, params, labels_gt, clean_pc in clouds:
        per_cloud.append(
            evaluate_cloud_dbscan(
                points,
                params,
                labels_gt,
                clean_pc,
                eps=float(eps),
                min_samples=int(min_samples),
                backend=backend,
            )
        )
    recall, specificity, gmean, mcc, wdist = _aggregate_cloud_metrics(per_cloud)
    return SplitMetrics(
        split=split,
        n_clouds=len(per_cloud),
        recall=recall,
        specificity=specificity,
        gmean=gmean,
        mcc=mcc,
        wdist=wdist,
        dbscan_eps=float(eps),
        dbscan_min_samples=int(min_samples),
        backend=backend,
    )


def load_run_config(run_dir: Path, base_config: str, seed: int | None) -> dict[str, Any]:
    manifest_path = run_dir / "logs" / "run_manifest.json"
    cfg = load_config(base_config, project_root=REPO_ROOT)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if seed is None:
            seed = manifest.get("seed")
    if seed is not None:
        cfg = deep_update(cfg, {"data": {"seed": int(seed)}})
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--base-config", type=str, default="reproduce")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--seed", type=int, default=None, help="Override data.seed (else run_manifest)")
    p.add_argument("--backend", type=str, default="ellphi", choices=["ellphi", "mahalanobis"])
    p.add_argument(
        "--dbscan-hparams",
        type=Path,
        default=None,
        help="JSON with eps/min_samples for test split",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write metrics JSON (default: run_dir/logs/)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir not found: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_run_config(run_dir, args.base_config, args.seed)

    eps = min_samples = None
    if args.split == "test":
        if args.dbscan_hparams is None:
            args.dbscan_hparams = run_dir / "logs" / "dbscan_hparams.json"
        hparams = json.loads(args.dbscan_hparams.read_text())
        eps = float(hparams["eps"])
        min_samples = int(hparams["min_samples"])

    metrics = evaluate_split(
        run_dir,
        config,
        args.split,
        device,
        eps=eps,
        min_samples=min_samples,
        backend=args.backend,
    )

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "val":
        hparams_path = log_dir / "dbscan_hparams.json"
        hparams_path.write_text(
            json.dumps(
                {
                    "eps": metrics.dbscan_eps,
                    "min_samples": metrics.dbscan_min_samples,
                    "backend": metrics.backend,
                    "mean_val_mcc_dbscan": metrics.mcc,
                    "selection": "max mean cloud MCC on validation split",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Saved DBSCAN hparams: {hparams_path}")

    out_path = args.out_json or log_dir / f"paper_metrics_{args.split}.json"
    payload = {
        "source_revision": git_revision(REPO_ROOT),
        "run_dir": str(run_dir),
        "split": args.split,
        **asdict(metrics),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
