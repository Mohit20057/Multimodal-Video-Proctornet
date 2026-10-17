"""
PyTorch Dataset and DataLoader for Proctoring Telemetry.
Streams from chunked HDF5 with on-the-fly spatial (ColorJitter, Blur)
and temporal (frame drop jitter) augmentations.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class ProctoringHDF5Dataset(Dataset):
    """
    Dataset streaming 16-frame video clips and 64x300 audio Mel-spectrograms from HDF5.
    Applies spatial and temporal augmentations during training.
    """
    def __init__(
        self,
        hdf5_path: str,
        split: str = 'train',
        is_training: bool = True
    ):
        super().__init__()
        import h5py
        self.hdf5_path = hdf5_path
        self.split = split
        self.is_training = is_training
        self.h5_file = None

        # Load indices for this split
        with h5py.File(self.hdf5_path, 'r') as h5f:
            idx_key = f"{split}_indices"
            if idx_key in h5f:
                self.indices = h5f[idx_key][:]
            else:
                total_samples = h5f['labels'].shape[0]
                self.indices = np.arange(total_samples)

    def _open_h5(self):
        import h5py
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open_h5()
        real_idx = self.indices[idx]

        video = self.h5_file['video'][real_idx]  # (16, 3, 112, 112)
        audio = self.h5_file['audio'][real_idx]  # (1, 64, 300)
        label = int(self.h5_file['labels'][real_idx])

        # Convert to float32 tensors
        video_t = torch.from_numpy(video).float()
        audio_t = torch.from_numpy(audio).float()
        label_t = torch.tensor(label, dtype=torch.long)

        # Apply data augmentations if training
        if self.is_training:
            video_t = self._apply_spatial_augmentations(video_t)
            video_t = self._apply_temporal_augmentations(video_t)

        return video_t, audio_t, label_t

    def _apply_spatial_augmentations(self, video: torch.Tensor) -> torch.Tensor:
        """
        Applies ColorJitter (brightness +-0.1, contrast +-0.1) and subtle Gaussian Blur.
        video: (16, 3, 112, 112) normalized in [-1.0, 1.0]
        """
        # Brightness adjustment
        delta_brightness = (torch.rand(1).item() * 0.2) - 0.1  # [-0.1, 0.1]
        video = torch.clamp(video + delta_brightness, -1.0, 1.0)

        # Contrast adjustment
        contrast_factor = 1.0 + ((torch.rand(1).item() * 0.2) - 0.1)  # [0.9, 1.1]
        mean = video.mean(dim=(-2, -1), keepdim=True)
        video = torch.clamp((video - mean) * contrast_factor + mean, -1.0, 1.0)

        # Gaussian blur (3x3 kernel) with 30% probability
        if torch.rand(1).item() < 0.3:
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
            kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            # Reshape (16, 3, 112, 112) -> (16*3, 1, 112, 112) for depthwise conv
            t_len, c, h, w = video.shape
            v_reshaped = video.view(t_len * c, 1, h, w)
            k_single = kernel[:1]
            blurred = torch.nn.functional.conv2d(v_reshaped, k_single, padding=1)
            video = blurred.view(t_len, c, h, w)

        return video

    def _apply_temporal_augmentations(self, video: torch.Tensor) -> torch.Tensor:
        """
        Simulates webcam packet jitter by randomly dropping and repeating +-2 frames.
        video: (16, 3, 112, 112)
        """
        if torch.rand(1).item() < 0.4:
            drop_count = torch.randint(1, 3, (1,)).item()
            drop_indices = torch.randperm(16)[:drop_count]
            # Replace dropped frames with neighboring frame
            for d in drop_indices:
                neighbor = max(0, d - 1) if d > 0 else min(15, d + 1)
                video[d] = video[neighbor]
        return video

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


def get_dataloaders(
    hdf5_path: str = "ml/data/proctoring_dataset.h5",
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates Train, Val, and Test DataLoaders strictly partitioned by Student ID.
    """
    train_dataset = ProctoringHDF5Dataset(hdf5_path, split='train', is_training=True)
    val_dataset = ProctoringHDF5Dataset(hdf5_path, split='val', is_training=False)
    test_dataset = ProctoringHDF5Dataset(hdf5_path, split='test', is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
