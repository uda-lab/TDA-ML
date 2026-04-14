import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import VietorisRipsComplex, WassersteinDistance
from src.topology import compute_anisotropic_distance_matrix

class TopologicalLoss(nn.Module):
    """
    Computes the Topological Loss between the predicted anisotropic filtration
    and the clean ground truth persistence diagram.

    Mathematical Formulation:
    1. Local Metric $M_i$: Defined by ellipse parameters (a_i, b_i, theta_i).
       $$M_i = R(theta_i) diag(a_i^-2, b_i^-2) R(theta_i)^T$$
    2. Anisotropic Distance $d_{M_i}$: Mahalanobis distance from x_i to x_j.
       $$d_{M_i}(x_i, x_j) = sqrt((x_j - x_i)^T M_i (x_j-x_i))$$
    3. Symmetrized Distance $D_{ij}$: Max-symmetrization to satisfy metric properties.
       $$D_{ij} = max { d_{M_i}(x_i, x_j), d_{M_j}(x_j, x_i) }$$
    4. Probability-Weighted Distance: Outlier penalty using inlier probabilities p_in.
       $$D'_{ij} = D_{ij} / sqrt(p_{in, i} * p_{in, j})$$
    5. Persistence Diagram Loss: Wasserstein distance W_2 of H1 diagrams.
       $$L_{topo} = W_2(PD_1(D'), PD_1(target))$$
    """
    def __init__(self, weight=0.1, epsilon=1e-8):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.vr_complex = VietorisRipsComplex(dim=1)
        self.wasserstein = WassersteinDistance(q=2)

    def forward(self, points, params, logits, clean_pd_info):
        """
        Args:
            points (torch.Tensor): (B, N, 2)
            params (torch.Tensor): (B, N, 5)
            logits (torch.Tensor): (B, N, 1) Outlier logits.
            clean_pd_info (list): List of clean persistence diagrams.
        """
        if self.weight <= 0:
            return torch.tensor(0.0, device=points.device)

        batch_size = points.shape[0]
        
        # 1-3. Compute Vectorized Anisotropic Distance Matrix
        # logits are outlier logits, so 1 - sigmoid(logits) are inlier probabilities.
        probs_outlier = torch.sigmoid(logits).squeeze(-1)
        
        # We use the vectorized helper in topology.py
        D_prime = compute_anisotropic_distance_matrix(
            points, params, probs=probs_outlier, symmetrize='max'
        )
        
        total_loss = 0.0
        valid_samples = 0
        
        # 5. Compute PH and Wasserstein Distance per sample
        # Note: torch-topological's forward is batched via batch_handler, 
        # but we need to match each pred PD with its specific clean PD.
        for i in range(batch_size):
            d_mat = D_prime[i]
            
            try:
                pd_pred_info = self.vr_complex(d_mat, treat_as_distances=True)
                
                # clean_pd_info[i] contains the target diagram
                loss_sample = self.wasserstein(pd_pred_info, clean_pd_info[i]) ** 2
                
                if not torch.isnan(loss_sample):
                    total_loss += loss_sample
                    valid_samples += 1
            except Exception as e:
                # Handle cases where Ripser might fail due to degenerate distances
                continue

        if valid_samples == 0:
            return torch.tensor(0.0, device=points.device, requires_grad=True)

        return self.weight * (total_loss / valid_samples)
