import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LensingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        class_map = {"no": 0, "sphere": 1, "vort": 2}

        for class_name, label in class_map.items():
            folder_path = os.path.join(image_dir, class_name)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith('.npy'):
                    self.images.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = np.load(img_path).astype(np.float32)

        if img.ndim == 2:
            img = img[:, :, np.newaxis]       
        elif img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)       


        if self.transform:
            img = self.transform(image=img)["image"]

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, label
