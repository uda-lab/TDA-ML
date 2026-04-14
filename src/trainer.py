import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance
from src.utils import sample_from_ellipses, visualize, calculate_persistent_entropy
from src.losses import TopologicalLoss
import os
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

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

        loss_cfg = config.get('loss', {})
        training_cfg = config.get('training', {})

        self.lambda_class = loss_cfg.get('w_class', training_cfg.get('lambda_class', 1.0))
        self.lambda_topo = loss_cfg.get('w_topo', training_cfg.get('lambda_topo', 0.1))
        self.lambda_aniso = loss_cfg.get('w_aniso', training_cfg.get('lambda_aniso', 0.01))
        
        self.lambda_major = training_cfg.get('lambda_major', training_cfg.get('lambda_size', 0.1))
        self.lambda_minor = training_cfg.get('lambda_minor', training_cfg.get('lambda_size', 0.1))
        
        self.lambda_min_b = loss_cfg.get('w_min_b', training_cfg.get('lambda_min_b', 0.0))
        self.min_b_target = training_cfg.get('min_b_target', 0.2)
        self.lambda_diversity = training_cfg.get('lambda_diversity', 0.0)
        
        self.aniso_mode = loss_cfg.get('aniso_mode', 'linear')
        print(f"Anisotropy Penalty Mode: {self.aniso_mode}")
        
        pos_weight_val = config.get('loss', {}).get('pos_weight', 1.0)
        if abs(pos_weight_val - 1.0) > 1e-6:
            print(f"Using pos_weight: {pos_weight_val}")
            self.pos_weight = torch.tensor([pos_weight_val], device=self.device)
            self.class_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.class_loss_fn = nn.BCEWithLogitsLoss()
        
        self.wasserstein = WassersteinDistance(q=2)

        self.visualize_every = config['training']['visualize_every']
        self.output_dir = config['outputs']['image_dir']
        self.log_dir = config['outputs']['log_dir']

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.topo_loss_fn = TopologicalLoss(weight=self.lambda_topo)
        self.fixed_indices = None
        self.threshold = config['model'].get('threshold', 0.5)

        self.vr_complex = VietorisRipsComplex(dim=1)
        
        self.topo_samples = training_cfg.get('topo_samples', 20)
        self.warmup_epochs = training_cfg.get('warmup_epochs', 0)

    def _compute_regularization_loss(self, params):
        """
        Compute regularization losses (Anisotropy, Size, MinB).
        params: (B, N, 5) [x, y, a, b, theta]
        """
        axes = params[:, :, 2:4]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        
        aniso_loss = torch.tensor(0.0, device=self.device)
        if abs(self.lambda_aniso) > 1e-9:
            aspect_ratios = major_axis / (minor_axis + 1e-6)
            
            if self.aniso_mode == 'barrier':
                barrier_threshold = self.config['training'].get('barrier_threshold', 6.0)
                barrier_term = F.relu(aspect_ratios - barrier_threshold).pow(2).mean()
                aniso_loss = 10.0 * barrier_term
            else:
                aniso_loss = aspect_ratios.mean()

        min_b_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_min_b > 0:
             min_b_loss = F.relu(self.min_b_target - minor_axis).mean()

        size_loss = (self.lambda_major * (major_axis**2) +
                     self.lambda_minor * (minor_axis**2)).mean()
             
        return aniso_loss, size_loss, min_b_loss

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_topo_loss = 0
        total_aniso_loss = 0
        total_size_loss = 0
        total_diversity_loss = 0
        total_entropy = 0
        total_entropy_samples = 0
        
        all_train_preds = []
        all_train_labels = []

        num_batches = len(data_loader)
        pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}")

        if self.fixed_indices is None:
            import random
            self.fixed_indices = random.sample(range(len(data_loader.dataset)), 3)

        for i, (data, labels, clean_pc) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.to(self.device)
            clean_pc = clean_pc.to(self.device)

            if self.config.get('training', {}).get('rotation_augmentation', False):
                theta = torch.rand(1, device=self.device) * 2 * 3.141592653589793
                cos_t, sin_t = torch.cos(theta), torch.sin(theta)
                rotation_matrix = torch.stack([
                    torch.stack([cos_t, -sin_t], dim=-1),
                    torch.stack([sin_t, cos_t], dim=-1)
                ], dim=-2).squeeze(0)
                
                data = torch.matmul(data, rotation_matrix.T)
                clean_pc = torch.matmul(clean_pc, rotation_matrix.T)

            self.optimizer.zero_grad()

            logits, params = self.model(data)

            class_loss = self.class_loss_fn(logits.squeeze(-1), labels.float())

            topo_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_topo > 0 and epoch > self.warmup_epochs:
                # Compute persistence diagrams for clean point clouds (Target)
                # Optimization: This could be precomputed in the dataset for even more speed.
                clean_pd_info = []
                for j in range(data.shape[0]):
                    c = clean_pc[j]
                    valid_mask = torch.abs(c).sum(dim=1) > 1e-6
                    clean_pd_info.append(self.vr_complex(c[valid_mask]))
                
                # Compute topological loss using the optimized class
                topo_loss = self.topo_loss_fn(data, params, logits, clean_pd_info)

            if self.lambda_topo > 0 and epoch <= self.warmup_epochs:
                 topo_loss = torch.tensor(0.0, device=self.device)

            if epoch > self.warmup_epochs:
                aniso_loss, size_loss, min_b_loss = self._compute_regularization_loss(params)
            else:
                aniso_loss = torch.tensor(0.0, device=self.device)
                size_loss = torch.tensor(0.0, device=self.device)
                min_b_loss = torch.tensor(0.0, device=self.device)

            diversity_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_diversity > 0 and epoch > self.warmup_epochs:
                angles = params[:, :, 4]
                angle_var = torch.var(angles, dim=1).mean()
                diversity_loss = -angle_var

            loss = class_loss + self.lambda_topo * topo_loss + self.lambda_aniso * aniso_loss + size_loss + self.lambda_min_b * min_b_loss + self.lambda_diversity * diversity_loss

            if torch.isnan(loss):
                print("Warning: NaN loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_value'])
            self.optimizer.step()

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_topo_loss += topo_loss.item()
            total_aniso_loss += aniso_loss.item()
            total_size_loss += size_loss.item()
            total_diversity_loss += diversity_loss.item()
            
            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs > self.threshold).long()
            all_train_preds.extend(preds.cpu().numpy().flatten())
            all_train_labels.extend(labels.cpu().numpy().flatten())

            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{class_loss.item():.4f}", topo=f"{topo_loss.item():.4f}", aniso=f"{aniso_loss.item():.4f}", size=f"{size_loss.item():.4f}", min_b=f"{min_b_loss.item():.4f}")
            if i % 10 == 0:
                print(f"Step {i}: Loss={loss.item():.4f}, Class={class_loss.item():.4f}, Topo={topo_loss.item():.4f}, Aniso={aniso_loss.item():.4f}, Size={size_loss.item():.4f}, MinB={min_b_loss.item():.4f}")

        avg_loss = total_loss / num_batches
        avg_class_loss = total_class_loss / num_batches
        avg_topo_loss = total_topo_loss / num_batches
        avg_aniso_loss = total_aniso_loss / num_batches
        avg_size_loss = total_size_loss / num_batches
        
        avg_entropy = total_entropy / total_entropy_samples if total_entropy_samples > 0 else 0.0
        
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(all_train_labels, all_train_preds, labels=[0, 1]).ravel()
        train_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        train_gmean = (train_recall * train_specificity) ** 0.5
        
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f} (Class: {avg_class_loss:.4f}, Topo: {avg_topo_loss:.4f}), Train F1: {train_f1:.4f}, Spec: {train_specificity:.4f}, G-Mean: {train_gmean:.4f}, MCC: {train_mcc:.4f}")

        if epoch % self.visualize_every == 0:
             visualize(self.model, self.device, data_loader.dataset, epoch, output_dir=self.output_dir, title_prefix=self.config['meta'].get('config_id', 'train'), sample_indices=self.fixed_indices, threshold=self.threshold)

        return avg_loss, avg_class_loss, avg_topo_loss, avg_aniso_loss, avg_size_loss, avg_entropy, train_f1, train_precision, train_recall, train_specificity, train_gmean, train_mcc

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for data, labels, _ in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                logits, params = self.model(data)
                class_loss = self.class_loss_fn(logits.squeeze(-1), labels.float())
                total_loss += class_loss.item()
                
                axes = params[:, :, 2:4]
                major_axis = axes.max(dim=-1)[0]
                minor_axis = axes.min(dim=-1)[0]
                aniso_loss = (1 - minor_axis / major_axis).mean()
                
                size_loss = (major_axis * minor_axis).mean()
                
                min_b_loss = F.relu(self.min_b_target - minor_axis).mean()

                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs > self.threshold).long()

                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                
                if not hasattr(self, 'val_aniso_accum'): self.val_aniso_accum = 0
                if not hasattr(self, 'val_size_accum'): self.val_size_accum = 0
                if not hasattr(self, 'val_min_b_accum'): self.val_min_b_accum = 0
                
                self.val_aniso_accum += aniso_loss.item()
                self.val_size_accum += size_loss.item()
                self.val_min_b_accum += min_b_loss.item()

        num_batches = len(data_loader) if len(data_loader) > 0 else 1
        avg_loss = total_loss / num_batches
        
        avg_aniso = getattr(self, 'val_aniso_accum', 0) / num_batches
        avg_size = getattr(self, 'val_size_accum', 0) / num_batches
        avg_min_b = getattr(self, 'val_min_b_accum', 0) / num_batches
        
        self.val_aniso_accum = 0
        self.val_size_accum = 0
        self.val_min_b_accum = 0

        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        gmean = (recall * specificity) ** 0.5
        
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return avg_loss, recall, specificity, gmean, mcc, avg_aniso, avg_size, avg_min_b

