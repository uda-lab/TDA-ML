import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np
from skimage.filters import threshold_otsu


class NoisyMNISTDataset(Dataset):
    """
    MNIST dataset converted to noisy 2D point clouds.

    Each image is binarized, converted to a set of (x, y) coordinates,
    subsampled or padded to a fixed size, perturbed with Gaussian noise,
    and augmented with uniformly random outlier points.

    Args:
        root (str): Path to store/load MNIST data.
        train (bool): Use training split if True, else test split.
        num_samples (int): Number of samples to use (randomly subsampled).
        max_points (int): Fixed number of inlier points per sample.
        num_outliers (int): Number of random outlier points to add.
        noise_std (float): Std of Gaussian jitter applied to inlier points.
        deterministic (bool): If True, fixes RNG per sample for reproducibility.
        indices (torch.Tensor, optional): Explicit index subset to use.
        noise_seed (int): Base seed for deterministic noise generation.

    Returns (per item):
        data (Tensor): Shape (max_points + num_outliers, 2). Shuffled point cloud.
        labels (Tensor): Shape (max_points + num_outliers,). 0=inlier, 1=outlier.
        clean_pc (Tensor): Shape (max_points, 2). Noise-free inlier points (zero-padded).
    """

    def __init__(self, root='./data', train=True, num_samples=5000,
                 max_points=150, num_outliers=20, noise_std=0.01,
                 deterministic=False, indices=None, noise_seed=0, preload=True):
        self.max_points = max_points
        self.num_outliers = num_outliers
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.noise_seed = noise_seed
        self.preload = preload

        full_dataset = datasets.MNIST(root, train=train, download=True)

        if indices is not None:
            self.images = full_dataset.data[indices]
            self.labels = full_dataset.targets[indices]
        elif num_samples < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:num_samples]
            self.images = full_dataset.data[indices]
            self.labels = full_dataset.targets[indices]
        else:
            self.images = full_dataset.data
            self.labels = full_dataset.targets

        self.preloaded_points = None
        if self.preload:
            self._preload_all_points()

    def _preload_all_points(self):
        """Pre-calculates the base point clouds for all images to speed up training."""
        print(f"Preloading {len(self.images)} MNIST samples into memory...")
        self.preloaded_points = []
        for i in range(len(self.images)):
            img = self.images[i].numpy()
            points = self._image_to_base_points(img)
            self.preloaded_points.append(points)
        print("Preloading complete.")

    def _image_to_base_points(self, img):
        """Converts a raw MNIST image to a normalized [-1, 1] point cloud."""
        try:
            thresh = threshold_otsu(img)
            binary_img = img > thresh
        except ValueError:
            binary_img = img > 0

        # np.argwhere returns (row, col) = (y, x)
        points = torch.tensor(np.argwhere(binary_img), dtype=torch.float32)
        if points.shape[0] > 0:
            # Map to Cartesian (x, y) and normalize to [-1, 1]
            # row -> 27-row (y), col -> x
            points = torch.stack([points[:, 1], 27 - points[:, 0]], dim=1)
            points = (points / 27.0) * 2.0 - 1.0
        return points

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a single sample and converts it to a mathematical point cloud.

        The process follows these mathematical steps:
        1. Binarization & Normalization (Preloaded if enabled)
        2. Subsampling/Padding to fixed size (max_points)
        3. Noise Injection & Outlier Addition
        """
        if self.preloaded_points is not None:
            points = self.preloaded_points[idx]
        else:
            img = self.images[idx].numpy()
            points = self._image_to_base_points(img)

        total_capacity = self.max_points + self.num_outliers

        if self.deterministic:
            rng = torch.Generator()
            rng.manual_seed(self.noise_seed + idx)
        else:
            rng = None

        num_points = points.shape[0]

        if num_points >= self.max_points:
            if self.deterministic:
                choice = torch.randperm(num_points, generator=rng)[:self.max_points]
            else:
                choice = torch.randperm(num_points)[:self.max_points]
            inliers = points[choice]
            clean_pc_points = points[choice]
        else:
            num_padding = self.max_points - num_points
            if self.deterministic:
                pad_indices = torch.randint(0, num_points, (num_padding,), generator=rng)
            else:
                pad_indices = torch.randint(0, num_points, (num_padding,))
            inliers = torch.cat([points, points[pad_indices]], dim=0)
            clean_pc_points = points

        if self.noise_std > 0:
            if self.deterministic:
                noise = torch.randn(inliers.size(), generator=rng) * self.noise_std
            else:
                noise = torch.randn_like(inliers) * self.noise_std
            inliers = inliers + noise

        if self.num_outliers > 0:
            if self.deterministic:
                outliers = torch.rand(self.num_outliers, 2, generator=rng) * 2.0 - 1.0
            else:
                outliers = torch.rand(self.num_outliers, 2) * 2.0 - 1.0
        else:
            outliers = torch.empty(0, 2)

        data = torch.zeros(total_capacity, 2)
        labels = torch.ones(total_capacity, dtype=torch.long)

        data[:self.max_points] = inliers
        labels[:self.max_points] = 0

        if self.num_outliers > 0:
            data[self.max_points:] = outliers

        if self.deterministic:
            perm = torch.randperm(total_capacity, generator=rng)
        else:
            perm = torch.randperm(total_capacity)
        data = data[perm]
        labels = labels[perm]

        clean_pc = torch.zeros(self.max_points, 2)
        clean_pc[:clean_pc_points.shape[0]] = clean_pc_points

        return data, labels, clean_pc


def create_data_loader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    """Wraps a dataset in a DataLoader with the given batch settings."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
