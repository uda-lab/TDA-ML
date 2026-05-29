import logging
import os

import torch
from torch.optim import Adam
from torch_topological.nn import VietorisRipsComplex
from tda_ml.visualization import visualize
from tda_ml.losses import (
    ClassificationLoss, 
    TopologicalLoss, 
    SizeRegularizationLoss, 
    AnisotropyPenaltyLoss
)
from tda_ml.metrics import compute_recall_specificity_gmean_mcc
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config, device=None, trial=None):
        self.model = model
        self.config = config
        self.trial = trial
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config['training']['lr'])
        training_cfg = config.get('training', {})
        perf_cfg = config.get('performance', {})
        self.use_amp = (
            self.device.type == "cuda"
            and bool(training_cfg.get("use_amp", perf_cfg.get("use_amp", True)))
        )
        self.autocast_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        amp_dtype_name = str(training_cfg.get("amp_dtype", perf_cfg.get("amp_dtype", "float16"))).lower()
        self.amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        loss_cfg = config.get('loss', {})

        self.lambda_class = loss_cfg.get('w_class', training_cfg.get('lambda_class', 1.0))
        self.lambda_topo = loss_cfg.get('w_topo', training_cfg.get('lambda_topo', 0.1))
        self.lambda_aniso = loss_cfg.get('w_aniso', training_cfg.get('lambda_aniso', 0.01))
        
        size_default = loss_cfg.get(
            "w_size", training_cfg.get("lambda_size", 0.1)
        )
        self.lambda_major = training_cfg.get("lambda_major", size_default)
        self.lambda_minor = training_cfg.get("lambda_minor", size_default)

        self.aniso_mode = loss_cfg.get("aniso_mode", training_cfg.get("aniso_mode", "linear"))
        logger.info("Anisotropy penalty mode: %s", self.aniso_mode)

        pos_weight_val = config.get('loss', {}).get('pos_weight', 1.0)
        pos_weight = torch.tensor([pos_weight_val], device=self.device) if pos_weight_val != 1.0 else None

        # Initialize Losses
        self.class_loss_fn = ClassificationLoss(pos_weight=pos_weight)
        _topo = config.get("model", {}).get("topology_loss", {})
        self.distance_backend = _topo.get("distance_backend", "mahalanobis")
        self.ellphi_differentiable = _topo.get("ellphi_differentiable", True)
        logger.info(
            "Topological distance backend: %s%s",
            self.distance_backend,
            (
                f" (ellphi_differentiable={self.ellphi_differentiable})"
                if self.distance_backend == "ellphi"
                else ""
            ),
        )
        self.topo_loss_fn = TopologicalLoss(
            weight=self.lambda_topo,
            distance_backend=self.distance_backend,
            ellphi_differentiable=self.ellphi_differentiable,
        )
        self.size_loss_fn = SizeRegularizationLoss(w_major=self.lambda_major, w_minor=self.lambda_minor)
        self.aniso_loss_fn = AnisotropyPenaltyLoss(
            weight=self.lambda_aniso, 
            mode=self.aniso_mode, 
            barrier_threshold=config['training'].get('barrier_threshold', 6.0)
        )

        self.visualize_every = config['training'].get('visualize_every', 5)
        self.output_dir = config['outputs']['image_dir']
        self.log_dir = config['outputs']['log_dir']

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.fixed_indices = None
        self.threshold = config['model'].get('threshold', 0.5)

        self.vr_complex = VietorisRipsComplex(dim=1)

        self.warmup_epochs = training_cfg.get('warmup_epochs', 0)

        self._val_aniso_accum = 0.0
        self._val_size_accum = 0.0

    # _compute_regularization_loss is now handled by classes in losses.py

    def _compute_clean_pd_info(self, clean_pc: torch.Tensor) -> list:
        """Compute clean persistence diagrams without gradient tracking."""
        clean_pd_info = []
        with torch.no_grad():
            for j in range(clean_pc.shape[0]):
                c = clean_pc[j]
                valid_mask = torch.abs(c).sum(dim=1) > 1e-6
                clean_pd_info.append(self.vr_complex(c[valid_mask]))
        return clean_pd_info

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_topo_loss = 0
        total_aniso_loss = 0
        total_size_loss = 0
        steps_completed = 0

        all_train_preds = []
        all_train_labels = []

        pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}")

        if self.fixed_indices is None:
            import random
            self.fixed_indices = random.sample(range(len(data_loader.dataset)), 3)

        for i, (data, labels, clean_pc) in enumerate(pbar):
            data = data.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            clean_pc = clean_pc.to(self.device, non_blocking=True)

            if self.config.get('training', {}).get('rotation_augmentation', False):
                theta = torch.rand(1, device=self.device) * 2 * 3.141592653589793
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                rotation_matrix = torch.stack([
                    torch.stack([cos_t, -sin_t], dim=-1),
                    torch.stack([sin_t, cos_t], dim=-1)
                ], dim=-2).squeeze(0)
                
                data = torch.matmul(data, rotation_matrix.T)
                clean_pc = torch.matmul(clean_pc, rotation_matrix.T)

            self.optimizer.zero_grad(set_to_none=True)

            clean_pd_info = None
            if self.lambda_topo > 0 and epoch > self.warmup_epochs:
                # Topological target PD does not require autograd; keep it out of AMP/grad graph.
                clean_pd_info = self._compute_clean_pd_info(clean_pc)

            with torch.amp.autocast(
                device_type=self.autocast_device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits, params = self.model(data)
                class_loss = self.class_loss_fn(logits, labels)

                topo_loss = torch.tensor(0.0, device=self.device)
                if clean_pd_info is not None:
                    topo_loss = self.topo_loss_fn(data, params, logits, clean_pd_info)

                # Regularization Losses (Size and Anisotropy only, as per slides)
                if epoch > self.warmup_epochs:
                    size_loss = self.size_loss_fn(params)
                    aniso_loss = self.aniso_loss_fn(params)
                else:
                    size_loss = torch.tensor(0.0, device=self.device)
                    aniso_loss = torch.tensor(0.0, device=self.device)

                loss = class_loss + topo_loss + aniso_loss + size_loss

            if torch.isnan(loss):
                logger.warning("NaN loss at epoch=%s batch_index=%s; skipping step", epoch, i)
                continue

            steps_completed += 1
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_value'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_value'])
                self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_topo_loss += topo_loss.item()
            total_aniso_loss += aniso_loss.item()
            total_size_loss += size_loss.item()
            
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > self.threshold).long()
            all_train_preds.extend(preds.cpu().numpy().flatten())
            all_train_labels.extend(labels.cpu().numpy().flatten())

            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{class_loss.item():.4f}", topo=f"{topo_loss.item():.4f}", aniso=f"{aniso_loss.item():.4f}", size=f"{size_loss.item():.4f}")
            if i % 10 == 0:
                logger.debug(
                    "Step %s: loss=%.4f class=%.4f topo=%.4f aniso=%.4f size=%.4f",
                    i,
                    loss.item(),
                    class_loss.item(),
                    topo_loss.item(),
                    aniso_loss.item(),
                    size_loss.item(),
                )

        if steps_completed == 0 or not all_train_labels:
            raise RuntimeError(
                f"All training batches were skipped at epoch={epoch}; loss was NaN for every batch."
            )

        denom = steps_completed if steps_completed > 0 else 1
        avg_loss = total_loss / denom
        avg_class_loss = total_class_loss / denom
        avg_topo_loss = total_topo_loss / denom
        avg_aniso_loss = total_aniso_loss / denom
        avg_size_loss = total_size_loss / denom

        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        
        _, train_specificity, train_gmean, train_mcc = compute_recall_specificity_gmean_mcc(
            all_train_labels, all_train_preds
        )
        
        logger.info(
            "Epoch %s avg loss=%.4f (class=%.4f topo=%.4f) train F1=%.4f spec=%.4f "
            "G-mean=%.4f MCC=%.4f",
            epoch,
            avg_loss,
            avg_class_loss,
            avg_topo_loss,
            train_f1,
            train_specificity,
            train_gmean,
            train_mcc,
        )

        if epoch % self.visualize_every == 0:
             visualize(self.model, self.device, data_loader.dataset, epoch, output_dir=self.output_dir, title_prefix=self.config['meta'].get('config_id', 'train'), sample_indices=self.fixed_indices, threshold=self.threshold)

        return (
            avg_loss,
            avg_class_loss,
            avg_topo_loss,
            avg_aniso_loss,
            avg_size_loss,
            train_f1,
            train_precision,
            train_recall,
            train_specificity,
            train_gmean,
            train_mcc,
        )

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []
        self._val_aniso_accum = 0.0
        self._val_size_accum = 0.0

        with torch.no_grad():
            for data, labels, _ in data_loader:
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=self.autocast_device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits, params = self.model(data)
                    class_loss = self.class_loss_fn(logits, labels)
                total_loss += class_loss.item()
                
                aniso_loss = self.aniso_loss_fn(params)
                size_loss = self.size_loss_fn(params)
                
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs > self.threshold).long()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                
                self._val_aniso_accum += aniso_loss.item()
                self._val_size_accum += size_loss.item()

        num_batches = len(data_loader) if len(data_loader) > 0 else 1
        avg_loss = total_loss / num_batches

        avg_aniso = self._val_aniso_accum / num_batches
        avg_size = self._val_size_accum / num_batches

        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        _, specificity, gmean, mcc = compute_recall_specificity_gmean_mcc(
            all_labels, all_preds
        )
        
        return avg_loss, recall, specificity, gmean, mcc, avg_aniso, avg_size

