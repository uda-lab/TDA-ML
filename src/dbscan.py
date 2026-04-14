import numpy as np
import torch
from sklearn.cluster import DBSCAN
from src.topology import compute_anisotropic_distance_matrix

def calculate_anisotropic_distance_matrix(points, params, metric='max', probs=None):
    """
    Calculate the anisotropic distance matrix between points.
    Uses the centralized logic from src.topology for mathematical consistency.

    Args:
        points (torch.Tensor or np.ndarray): (N, 2) coordinates.
        params (torch.Tensor or np.ndarray): (N, 5) ellipse parameters (x, y, a, b, theta).
        metric (str): Symmetrization strategy. 'max' or 'min'. Default 'max'.
        probs (torch.Tensor or np.ndarray, optional): (N,) Outlier probabilities. 
                                                      Distance is scaled by 1/(1-prob).
        
    Returns:
        np.ndarray: (N, N) distance matrix.
    """
    # Convert inputs to torch tensors for centralized computation
    points_t = torch.as_tensor(points, dtype=torch.float32)
    params_t = torch.as_tensor(params, dtype=torch.float32)
    probs_t = torch.as_tensor(probs, dtype=torch.float32) if probs is not None else None
    
    # Ensure Batch dimension (1, N, D)
    if points_t.ndim == 2:
        points_t = points_t.unsqueeze(0)
        params_t = params_t.unsqueeze(0)
        if probs_t is not None:
            probs_t = probs_t.unsqueeze(0)
            
    # Compute using vectorized torch logic
    with torch.no_grad():
        dist_matrix_t = compute_anisotropic_distance_matrix(
            points_t, params_t, probs=probs_t, symmetrize=metric
        )
    
    # Convert back to numpy for sklearn
    return dist_matrix_t.squeeze(0).cpu().numpy()

def apply_anisotropic_dbscan(points, params, eps=0.5, min_samples=5, metric='max', probs=None):
    """
    Apply DBSCAN using anisotropic distance.
    
    Args:
        points: (N, 2) coordinates
        params: (N, 5) ellipse parameters
        eps: DBSCAN epsilon
        min_samples: DBSCAN min_samples
        metric: Symmetrization strategy ('max' or 'min')
        probs: (N,) Outlier probabilities for weighting distance
        
    Returns:
        labels: Cluster labels for each point (-1 for noise)
    """
    dist_matrix = calculate_anisotropic_distance_matrix(points, params, metric=metric, probs=probs)
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    
    return labels
