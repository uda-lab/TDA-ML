import torch
import numpy as np

def compute_anisotropic_metric(axes, angles):
    """
    Computes the components of the local metric tensor M_i based on ellipse parameters.
    
    Mathematical Definition:
    M_i = R(theta_i) diag(a_i^-2, b_i^-2) R(theta_i)^T
    
    Args:
        axes (torch.Tensor): (..., 2) [a, b]
        angles (torch.Tensor): (..., 1) [theta]
        
    Returns:
        tuple: (m00, m11, m01) components of the metric tensors.
    """
    a = axes[..., 0] + 1e-6
    b = axes[..., 1] + 1e-6
    theta = angles.squeeze(-1)
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    inv_a2 = 1.0 / (a**2 + 1e-8)
    inv_b2 = 1.0 / (b**2 + 1e-8)
    
    # Components of M = R diag(a^-2, b^-2) R^T
    m00 = cos_t**2 * inv_a2 + sin_t**2 * inv_b2
    m11 = sin_t**2 * inv_a2 + cos_t**2 * inv_b2
    m01 = cos_t * sin_t * (inv_a2 - inv_b2)
    
    return m00, m11, m01

def compute_anisotropic_distance_matrix(points, params, probs=None, symmetrize='max'):
    """
    Computes the anisotropic distance matrix between points in a vectorized manner.
    
    Mathematical Formulation:
    1. Distance from $x_i$ to $x_j$ under metric $M_i$: 
       $d_{M_i}(x_i, x_j) = \sqrt{(x_j - x_i)^\top M_i (x_j - x_i)}$
    2. Symmetrized Distance $D_{ij}$: 
       $D_{ij} = \max \{ d_{M_i}(x_i, x_j), d_{M_j}(x_j, x_i) \}$
    3. Probability Weighting $p_{in, i}$:
       $D'_{ij} = D_{ij} / p_{in, i}$ (Asymmetric) or $D_{ij} / \sqrt{p_{in, i} p_{in, j}}$ (Symmetric)
       Default: Symmetric $D'_{ij} = D_{ij} / \sqrt{p_{in, i} p_{in, j}}$
    
    Args:
        points (torch.Tensor): (B, N, 2) Coordinates of centers.
        params (torch.Tensor): (B, N, 5) Ellipse parameters [dx, dy, a, b, theta].
        probs (torch.Tensor, optional): (B, N) Outlier probabilities. 
                                     If provided, distances are weighted by 1/(inlier_prob).
        symmetrize (str): 'max' or 'min' for symmetrization. Default 'max'.
        
    Returns:
        torch.Tensor: (B, N, N) Symmetric distance matrices.
    """
    B, N, _ = points.shape
    device = points.device
    
    # Extract params
    # Note: params[..., 0:2] are offsets, but distance is between centers x_i and x_j.
    # The presenter define distance as between points centers.
    centers = points + params[..., 0:2]
    m00, m11, m01 = compute_anisotropic_metric(params[..., 2:4], params[..., 4:5])
    
    # Compute relative coordinates (B, N, N, 2)
    # diff[b, i, j] = centers[b, j] - centers[b, i]
    diff = centers.unsqueeze(2) - centers.unsqueeze(1)
    dx = diff[..., 0]
    dy = diff[..., 1]
    
    # Compute d_Mi(xi, xj)^2 for all i, j
    # m00 shape is (B, N), need to broadcast to (B, N, N)
    dist_sq_ij = (m00.unsqueeze(2) * dx**2 +
                  2 * m01.unsqueeze(2) * dx * dy +
                  m11.unsqueeze(2) * dy**2)
    
    # Weight by inlier probabilities if provided.
    # Logic: D_ij = D_ij / sqrt(p_i * p_j)  =>  D_ij^2 = D_ij^2 / (p_i * p_j)
    if probs is not None:
        inlier_probs = 1.0 - probs
        inlier_probs = torch.clamp(inlier_probs, min=1e-4)
        prob_matrix = inlier_probs.unsqueeze(2) * inlier_probs.unsqueeze(1) # (B, N, N)
        dist_sq_ij = dist_sq_ij / (prob_matrix + 1e-8)
        
    # Symmetrization
    dist_sq_ji = dist_sq_ij.transpose(-1, -2)
    
    if symmetrize == 'max':
        dist_sq = torch.max(dist_sq_ij, dist_sq_ji)
    elif symmetrize == 'min':
        dist_sq = torch.min(dist_sq_ij, dist_sq_ji)
    else:
        dist_sq = dist_sq_ij
        
    return torch.sqrt(torch.clamp(dist_sq, min=1e-8))
