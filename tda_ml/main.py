import argparse
import csv
import json
import logging
import os

import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import deep_update, load_config, model_kwargs_from_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.seed_utils import set_global_seed
from tda_ml.runtime_profile import build_runtime_profile
from tda_ml.supervised_diagnostics import (
    git_revision,
    run_abort_diagnostics,
    should_early_abort,
    write_abort_report,
)
from tda_ml.trainer import Trainer

logger = logging.getLogger(__name__)


def _resolve_dataloader_settings(config, device):
    data_cfg = config["data"]
    cpu_threads = os.cpu_count() or 8
    default_workers = min(16, max(4, cpu_threads // 2))

    num_workers = int(data_cfg.get("num_workers", default_workers))
    num_workers = max(0, min(num_workers, cpu_threads))

    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)

    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None

    return num_workers, pin_memory, persistent_workers, prefetch_factor


def _configure_torch_runtime(config, device):
    perf_cfg = config.get("performance", {})
    if device.type != "cuda":
        return

    if perf_cfg.get("enable_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(perf_cfg.get("matmul_precision", "high"))
    torch.backends.cudnn.benchmark = bool(perf_cfg.get("cudnn_benchmark", True))

def main(config_name=None, config=None, trial=None, config_overrides=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if config is None:
        if config_name is None:
            raise ValueError("Either config_name or config must be provided")
        config = load_config(config_name)
        
    if config_overrides:
        deep_update(config, config_overrides)
    
    logger.info("Loaded config: %s", config["meta"].get("config_id", "unknown"))

    if config.get('device') and config['device'] != 'auto':
        device = torch.device(config['device'])
    elif torch.cuda.is_available():
        device = torch.device('cuda') 
    elif torch.backends.mps.is_available():
        device = torch.device('mps') 
    else:
        device = torch.device('cpu')

    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_id = config['meta'].get('config_id', 'unknown')
    
    # --- Directory Setup ---
    if 'outputs' in config:
        base_dir = config['outputs'].get('base_dir', 'outputs')
        run_dir_name = f"{config_id}_{timestamp}"
        run_dir = os.path.join(base_dir, run_dir_name)
        
        # Consistent directory names across the project
        config['outputs']['log_dir'] = os.path.join(run_dir, 'logs')
        config['outputs']['image_dir'] = os.path.join(run_dir, 'images')
        
        # Note: Trainer also calls makedirs to ensure robustness
        os.makedirs(config['outputs']['log_dir'], exist_ok=True)
        os.makedirs(config['outputs']['image_dir'], exist_ok=True)
        
        logger.info("Project structure created at: %s", run_dir)

    data_cfg = config['data']
    seed = data_cfg.get('seed', 42)
    deterministic_algorithms = bool(config.get("reproducibility", {}).get("deterministic_algorithms", False))
    set_global_seed(seed, deterministic_algorithms=deterministic_algorithms)
    logger.info(
        "Global seed initialized: seed=%s, deterministic_algorithms=%s",
        seed,
        deterministic_algorithms,
    )
    train_size = data_cfg.get('train_size', 4500)
    val_size = data_cfg.get('val_size', 500)
    test_size = data_cfg.get('test_size', 1000)
    
    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[:train_size + val_size]
    
    train_indices = full_train_indices[:train_size]
    val_indices = full_train_indices[train_size:]
    
    num_workers, use_pin_memory, persistent_workers, prefetch_factor = _resolve_dataloader_settings(config, device)
    _configure_torch_runtime(config, device)
    logger.info(
        "DataLoader settings: workers=%s, pin_memory=%s, persistent_workers=%s, prefetch_factor=%s",
        num_workers,
        use_pin_memory,
        persistent_workers,
        prefetch_factor,
    )

    train_dataset = NoisyMNISTDataset(
        root='./data', train=True, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=train_indices,
        deterministic=True, noise_seed=seed
    )
    
    data_loader = create_data_loader(
        train_dataset, 
        batch_size=data_cfg['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    val_dataset = NoisyMNISTDataset(
        root='./data', train=True, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=val_indices,
        deterministic=True, noise_seed=seed
    )
    
    val_loader = create_data_loader(
        val_dataset, 
        batch_size=data_cfg['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_indices = torch.randperm(10000, generator=generator)[:test_size]
    test_dataset = NoisyMNISTDataset(
        root='./data', train=False, max_points=data_cfg['max_points'],
        num_outliers=data_cfg['num_outliers'], indices=test_indices,
        deterministic=True, noise_seed=seed
    )
    
    test_loader = create_data_loader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = AnisotropicOutlierClassifier(**model_kwargs_from_config(config))
    model.to(device)

    trainer = Trainer(model, config, device=device) 

    init_checkpoint = config.get('init_checkpoint')
    if init_checkpoint and os.path.exists(init_checkpoint):
        logger.info("Loading initial weights from %s", init_checkpoint)
        checkpoint = load_torch_checkpoint(init_checkpoint, map_location=device)
        model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    elif init_checkpoint:
        logger.warning("Initial checkpoint not found at %s", init_checkpoint)

    log_dir = config['outputs']['log_dir']
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    runtime_profile = build_runtime_profile(
        config=config,
        device=device,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_amp_effective=trainer.use_amp,
        amp_dtype_effective=str(trainer.amp_dtype).replace("torch.", ""),
    )
    runtime_profile_path = os.path.join(log_dir, "runtime_profile.json")
    with open(runtime_profile_path, "w", encoding="utf-8") as f:
        json.dump(runtime_profile, f, ensure_ascii=True, indent=2)
    logger.info("Runtime profile saved: %s", runtime_profile_path)

    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_revision": git_revision(),
        "config_id": config_id,
        "command_entry": "tda_ml.main",
        "seed": seed,
        "epochs_planned": config["training"]["epochs"],
        "distance_backend": config.get("model", {})
        .get("topology_loss", {})
        .get("distance_backend", "mahalanobis"),
        "early_abort": config.get("training", {}).get("early_abort"),
        "run_dir": run_dir,
        "fallback_status": "not applicable",
    }
    manifest_path = os.path.join(log_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
    logger.info("Run manifest saved: %s", manifest_path)

    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_class_loss', 'train_topo_loss', 'train_aniso_loss', 'train_size_loss', 'val_loss', 'val_recall', 'val_mcc', 'val_aniso', 'val_size'])

    # --- Training Loop ---
    epochs = config['training']['epochs']
    save_every = config['outputs'].get('save_every', 10)
    best_val_mcc = -1.0
    early_abort_cfg = config.get("training", {}).get("early_abort", {})
    metrics_history: list[dict] = []
    final_status = "completed"
    abort_report_path = None

    for epoch in range(1, epochs + 1):
        # res returns (avg_loss, class_loss, topo_loss, aniso_loss, size_loss, ...)
        res = trainer.train_epoch(data_loader, epoch)
        val_res = trainer.validate(val_loader)

        val_mcc = val_res[4] # MCC is at index 4
        train_mcc = res[10]
        val_recall = val_res[1]
        print(f"Epoch {epoch}: Val MCC={val_mcc:.4f}, Aniso={val_res[5]:.4f}")

        metrics_history.append(
            {
                "epoch": epoch,
                "val_mcc": float(val_mcc),
                "train_mcc": float(train_mcc),
                "val_recall": float(val_recall),
                "val_specificity": float(val_res[2]),
                "val_loss": float(val_res[0]),
                "train_loss": float(res[0]),
                "val_size": float(val_res[6]),
                "val_aniso": float(val_res[5]),
            }
        )

        # Save best model logic
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_mcc,
            }, best_model_path)
            print(f"Saved best model (MCC: {val_mcc:.4f}) to {best_model_path}")

        if epoch % save_every == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mcc': val_mcc,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, res[0], res[1], res[2], res[3], res[4], val_res[0], val_res[1], val_res[4], val_res[5], val_res[6]])

        do_abort, abort_reason = should_early_abort(
            epoch=epoch,
            best_val_mcc=best_val_mcc,
            val_recall=val_recall,
            val_mcc=val_mcc,
            train_mcc=train_mcc,
            early_abort_cfg=early_abort_cfg,
        )
        if do_abort:
            abort_ckpt = os.path.join(run_dir, "abort_checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mcc": val_mcc,
                    "best_val_mcc": best_val_mcc,
                    "abort_reason": abort_reason,
                },
                abort_ckpt,
            )
            logger.warning("Early abort at epoch %s: %s", epoch, abort_reason)
            report = run_abort_diagnostics(
                trainer=trainer,
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                metrics_history=metrics_history,
                abort_reason=abort_reason,
            )
            abort_report_path = write_abort_report(log_dir, report)
            logger.warning("Abort diagnostics: %s", abort_report_path)
            manifest["final_status"] = "early-aborted"
            manifest["abort_epoch"] = epoch
            manifest["abort_reason"] = abort_reason
            manifest["abort_report"] = str(abort_report_path)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
            final_status = "early-aborted"
            break

    if final_status == "completed":
        manifest["final_status"] = "completed"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=True, indent=2)

    if final_status == "early-aborted":
        return {
            "val_gmean": val_res[3],
            "val_mcc": val_res[4],
            "run_dir": run_dir,
            "status": final_status,
            "best_val_mcc": best_val_mcc,
            "abort_report": str(abort_report_path) if abort_report_path else None,
        }

    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    test_res = trainer.validate(test_loader)
    (
        test_loss,
        test_recall,
        test_specificity,
        test_gmean,
        test_mcc,
        test_aniso,
        test_size,
    ) = test_res
    test_metrics = {
        "test_loss": test_loss,
        "test_recall": test_recall,
        "test_specificity": test_specificity,
        "test_gmean": test_gmean,
        "test_mcc": test_mcc,
        "test_aniso": test_aniso,
        "test_size": test_size,
    }
    test_metrics_path = os.path.join(log_dir, "test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=True, indent=2)
    logger.info(
        "Test (held-out): MCC=%.4f, recall=%.4f, specificity=%.4f, G-mean=%.4f, loss=%.4f",
        test_mcc,
        test_recall,
        test_specificity,
        test_gmean,
        test_loss,
    )
    logger.info("Test metrics saved: %s", test_metrics_path)

    return {
        "val_gmean": val_res[3],
        "val_mcc": val_res[4],
        "run_dir": run_dir,
        "status": final_status,
        "best_val_mcc": best_val_mcc,
        "abort_report": None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='dev', help='Config path')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='Path to initial checkpoint')
    args = parser.parse_args()
    
    config_overrides = {}
    if args.init_checkpoint:
        config_overrides['init_checkpoint'] = args.init_checkpoint
        
    main(config_name=args.config, config_overrides=config_overrides) 
