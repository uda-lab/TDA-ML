import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance
from src.topology import compute_anisotropic_distance_matrix

class ClassificationLoss(nn.Module):
    """
    Standard Binary Cross-Entropy with Logits for inlier/outlier classification.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, labels):
        return self.loss_fn(logits.squeeze(-1), labels.float())

class SizeRegularizationLoss(nn.Module):
    """
    Penalizes the size of estimated ellipses to prevent over-expansion.
    
    Formula: L = lambda_major * a^2 + lambda_minor * b^2
    """
    def __init__(self, w_major=0.1, w_minor=0.1):
        super().__init__()
        self.w_major = w_major
        self.w_minor = w_minor

    def forward(self, params):
        axes = params[..., 2:4]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        loss = (self.w_major * (major_axis**2) + self.w_minor * (minor_axis**2)).mean()
        return loss

class AnisotropyPenaltyLoss(nn.Module):
    """
    Prevents ellipses from becoming too elongated by penalizing high aspect ratios.
    
    Modes:
    - linear: Penalizes aspect ratio (major/minor) directly.
    - barrier: Penalizes aspect ratio squared only above a certain threshold.
    """
    def __init__(self, weight=0.01, mode='linear', barrier_threshold=6.0):
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.barrier_threshold = barrier_threshold

    def forward(self, params):
        if abs(self.weight) < 1e-9:
            return torch.tensor(0.0, device=params.device)
            
        axes = params[..., 2:4]
        major_axis = axes.max(dim=-1)[0]
        minor_axis = axes.min(dim=-1)[0]
        
        aspect_ratios = major_axis / (minor_axis + 1e-6)
        
        if self.mode == 'barrier':
            barrier_term = F.relu(aspect_ratios - self.barrier_threshold).pow(2).mean()
            loss = 10.0 * barrier_term
        else:
            loss = aspect_ratios.mean()
            
        return self.weight * loss

class TopologicalLoss(nn.Module):
    """
    Computes the Topological Loss between the predicted anisotropic filtration
    and the clean ground truth persistence diagram using Wasserstein distance.
    """
    def __init__(self, weight=0.1, epsilon=1e-8):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.vr_complex = VietorisRipsComplex(dim=1)
        self.wasserstein = WassersteinDistance(q=2)

    def forward(self, points, params, logits, clean_pd_info):
        if self.weight <= 0:
            return torch.tensor(0.0, device=points.device)

        batch_size = points.shape[0]
        probs_outlier = torch.sigmoid(logits).squeeze(-1)
        
        # Compute Vectorized Anisotropic Distance Matrix
        D_prime = compute_anisotropic_distance_matrix(
            points, params, probs=probs_outlier, symmetrize='max'
        )
        
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            d_mat = D_prime[i]
            try:
                pd_pred_info = self.vr_complex(d_mat, treat_as_distances=True)
                loss_sample = self.wasserstein(pd_pred_info, clean_pd_info[i]) ** 2
                
                if not torch.isnan(loss_sample):
                    total_loss += loss_sample
                    valid_samples += 1
            except Exception:
                continue

        if valid_samples == 0:
            return torch.tensor(0.0, device=points.device, requires_grad=True)

        return self.weight * (total_loss / valid_samples)
