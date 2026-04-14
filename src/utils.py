import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import gudhi
import gudhi.wasserstein

def compute_w_distance(points_pred, points_gt):
    """
    Computes Wasserstein distance between persistence diagrams (dim 1) of two point clouds.
    Args:
        points_pred: (N, 2) array or tensor
        points_gt: (M, 2) array or tensor
    Returns:
        distance: float
    """
    if isinstance(points_pred, torch.Tensor):
        points_pred = points_pred.detach().cpu().numpy()
    if isinstance(points_gt, torch.Tensor):
        points_gt = points_gt.detach().cpu().numpy()
        
    if len(points_pred) == 0:
        return 999.0
    if len(points_gt) == 0:
        return 0.0
        
    def get_pd(pts):
        if len(pts) < 3:
            return []
        alpha_complex = gudhi.AlphaComplex(points=pts)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.compute_persistence()
        pd = simplex_tree.persistence_intervals_in_dimension(1)
        return pd

    pd_pred = get_pd(points_pred)
    pd_gt = get_pd(points_gt)
    
    if len(pd_pred) == 0 and len(pd_gt) == 0:
        return 0.0
    if len(pd_pred) == 0:
        pd_pred = np.empty((0, 2))
    if len(pd_gt) == 0:
        pd_gt = np.empty((0, 2))
        
    dist = gudhi.wasserstein.wasserstein_distance(pd_pred, pd_gt, order=1, internal_p=2)
    return dist

def compute_bottleneck_distance(points_pred, points_gt):
    """
    Computes Bottleneck distance between persistence diagrams (dim 1).
    """
    if isinstance(points_pred, torch.Tensor):
        points_pred = points_pred.detach().cpu().numpy()
    if isinstance(points_gt, torch.Tensor):
        points_gt = points_gt.detach().cpu().numpy()
        
    if len(points_pred) == 0:
        return 99.0
    if len(points_gt) == 0:
        return 0.0

    def get_pd(pts):
        if len(pts) < 3:
            return []
        alpha_complex = gudhi.AlphaComplex(points=pts)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.compute_persistence()
        pd = simplex_tree.persistence_intervals_in_dimension(1)
        return pd

    pd_pred = get_pd(points_pred)
    pd_gt = get_pd(points_gt)
    
    if len(pd_pred) == 0 and len(pd_gt) == 0:
        return 0.0
    if len(pd_pred) == 0:
        pd_pred = np.empty((0, 2))
    if len(pd_gt) == 0:
        pd_gt = np.empty((0, 2))
        
    dist = gudhi.bottleneck_distance(pd_pred, pd_gt)
    return dist

def compute_betti_error(points_pred, points_gt):
    """
    Computes absolute error in Betti numbers (dim 0 and 1).
    Returns: (betti0_err, betti1_err)
    """
    if isinstance(points_pred, torch.Tensor):
        points_pred = points_pred.detach().cpu().numpy()
    if isinstance(points_gt, torch.Tensor):
        points_gt = points_gt.detach().cpu().numpy()
        
    def get_betti(pts):
        if len(pts) == 0:
            return 0, 0
        if len(pts) < 3:
            return 1, 0
        alpha_complex = gudhi.AlphaComplex(points=pts)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.compute_persistence()
        betti = simplex_tree.betti_numbers()
        b0 = betti[0] if len(betti) > 0 else 0
        b1 = betti[1] if len(betti) > 1 else 0
        return b0, b1

    b0_pred, b1_pred = get_betti(points_pred)
    b0_gt, b1_gt = get_betti(points_gt)
    
    return abs(b0_pred - b0_gt), abs(b1_pred - b1_gt)

def sample_from_ellipses(points, ellipse_params, num_samples_per_ellipse=50):
    """
    Samples points from the predicted ellipses.
    Args:
        points: (B, N, 2)
        ellipse_params: (B, N, 5) [dx, dy, a, b, theta]
    Returns:
        sampled_points: (B, N * num_samples, 2)
    """
    batch_size, N, _ = ellipse_params.shape
    device = ellipse_params.device

    cx = points[..., 0:1] + ellipse_params[..., 0:1]
    cy = points[..., 1:2] + ellipse_params[..., 1:2]

    a = ellipse_params[..., 2:3]
    b = ellipse_params[..., 3:4]
    theta = ellipse_params[..., 4:5]

    t = torch.linspace(0, 2 * torch.pi, num_samples_per_ellipse, device=device).view(1, 1, num_samples_per_ellipse)

    a = a.unsqueeze(-1)
    b = b.unsqueeze(-1)
    theta = theta.unsqueeze(-1)
    cx = cx.unsqueeze(-1)
    cy = cy.unsqueeze(-1)

    x_e = a * torch.cos(t)
    y_e = b * torch.sin(t)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    x_r = x_e * cos_theta - y_e * sin_theta + cx
    y_r = x_e * sin_theta + y_e * cos_theta + cy

    sampled_points = torch.stack([x_r, y_r], dim=-1)
    return sampled_points.view(batch_size, N * num_samples_per_ellipse, 2)

def calculate_persistent_entropy(pd_info):
    """
    Calculates persistent entropy from persistence information.
    Args:
        pd_info: List of persistence diagrams [dim0, dim1, ...]
    Returns:
        entropy: float
    """
    if len(pd_info) < 2:
        return 0.0
    
    pd_h1 = pd_info[1]
    
    if not isinstance(pd_h1, torch.Tensor):
         if hasattr(pd_h1, 'diagram'):
             pd_h1 = pd_h1.diagram
    
    if pd_h1.size(0) == 0:
        return 0.0

    births = pd_h1[:, 0]
    deaths = pd_h1[:, 1]
    
    finite_mask = torch.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]
    
    if len(births) == 0:
        return 0.0

    lifetimes = deaths - births
    
    valid_mask = lifetimes > 1e-6
    lifetimes = lifetimes[valid_mask]
    
    if len(lifetimes) == 0:
        return 0.0

    total_lifetime = lifetimes.sum()
    probabilities = lifetimes / total_lifetime
    
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    
    return entropy.item()

def visualize(model, device, dataset, epoch, output_dir=".", title_prefix="", sample_indices=None, threshold=0.5):
    """
    Visualizes model predictions on 3 random samples from the dataset.
    """
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Epoch {epoch} - {title_prefix} Results', fontsize=16)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(3):
            if sample_indices is not None and i < len(sample_indices):
                idx = sample_indices[i]
            else:
                idx = torch.randint(0, len(dataset), (1,)).item()
            data, labels, clean_pc = dataset[idx]

            data_np = data.numpy()
            labels_np = labels.numpy()

            data_batch = data.to(device).unsqueeze(0)
            logits, params = model(data_batch)

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            pred_labels = (probs > threshold).astype(int).flatten()

            params_np = params.squeeze(0).cpu().numpy()
            
            # Convert specifically to torch to use centralized math logic
            axes_t = params[:, :, 2:4]
            angles_t = params[:, :, 4:5]
            from src.topology import compute_anisotropic_metric
            m00, m11, m01 = compute_anisotropic_metric(axes_t, angles_t)
            m00_np = m00.squeeze(0).cpu().numpy()
            m11_np = m11.squeeze(0).cpu().numpy()
            m01_np = m01.squeeze(0).cpu().numpy()

            axes[i, 0].scatter(data_np[labels_np==1, 0], data_np[labels_np==1, 1], c='red', s=10, label='Outlier (GT)')
            axes[i, 0].scatter(data_np[labels_np==0, 0], data_np[labels_np==0, 1], c='blue', s=10, label='Inlier (GT)')
            axes[i, 0].set_title(f'Sample {i+1}: GT Labels')
            axes[i, 0].legend()

            axes[i, 1].scatter(data_np[pred_labels==1, 0], data_np[pred_labels==1, 1], c='red', s=10, marker='x', label='Pred Outlier')
            axes[i, 1].scatter(data_np[pred_labels==0, 0], data_np[pred_labels==0, 1], c='blue', s=10, label='Pred Inlier')
            axes[i, 1].set_title(f'Sample {i+1}: Prediction')
            axes[i, 1].legend()

            axes[i, 2].scatter(data_np[pred_labels==1, 0], data_np[pred_labels==1, 1], c='red', s=10, marker='x', label='Pred Outlier')
            axes[i, 2].scatter(data_np[pred_labels==0, 0], data_np[pred_labels==0, 1], c='blue', s=10, label='Pred Inlier')

            t = np.linspace(0, 2*np.pi, 50)
            if len(data_np) > 0:
                for k in range(len(data_np)):
                    dx, dy, a, b, theta = params_np[k]
                    
                    cx = data_np[k, 0] + dx
                    cy = data_np[k, 1] + dy
                    
                    # Parametric equation derived from centered matrix Mi
                    x_e = a * np.cos(t)
                    y_e = b * np.sin(t)
                    x_r = x_e * np.cos(theta) - y_e * np.sin(theta) + cx
                    y_r = x_e * np.sin(theta) + y_e * np.cos(theta) + cy
                    
                    color = 'g-' if pred_labels[k] == 0 else 'r-'
                    axes[i, 2].plot(x_r, y_r, color, alpha=0.3, linewidth=1)

            axes[i, 2].set_title(f'Sample {i+1}: Predicted Inliers & Ellipses')
            for ax in axes[i]:
                ax.set_aspect('equal')
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = os.path.join(output_dir, f"{title_prefix}_result_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()

import yaml

def load_config(config_name):
    """
    Loads configuration by merging base.yaml with the environment specific yaml.
    Args:
        config_name (str): 'dev' or 'prod' (or path to yaml)
    """
    with open('configs/base.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    if config_name.endswith('.yaml'):
        env_config_path = config_name
    else:
        env_config_path = f"configs/{config_name}.yaml"

    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = update(base_config, env_config)
    return config
