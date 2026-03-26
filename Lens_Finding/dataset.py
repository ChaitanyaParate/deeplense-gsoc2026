import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LensDataset(Dataset):
    def __init__(self, lens_dir, nonlens_dir, transform=None):
        self.transform = transform
        lens_files = [(os.path.join(lens_dir, f), 1)
                      for f in os.listdir(lens_dir) if f.endswith('.npy')]
        nonlens_files = [(os.path.join(nonlens_dir, f), 0)
                         for f in os.listdir(nonlens_dir) if f.endswith('.npy')]
        self.data = lens_files + nonlens_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = np.load(path).astype(np.float32)

        # Clip BEFORE normalizing
        img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))

        for c in range(img.shape[0]):
            mn, mx = img[c].min(), img[c].max()
            if mx > mn:
                img[c] = (img[c] - mn) / (mx - mn)

        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
