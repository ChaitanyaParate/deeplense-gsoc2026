from torchvision import transforms
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from dataset import LensDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import os


def save_checkpoint(state, filename="vit_lensing_best.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None):
    print("=> Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print("Checkpoint loaded successfully!")

def get_loaders(data_dir, train_transform, val_transform, batch_size, num_workers, pin_memory):

    cache_path = os.path.join(data_dir, "dataset_stats.npz")

    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        mean, std = cache["mean"].tolist(), cache["std"].tolist()
        print(f"Loaded cached stats - Mean: {mean}, Std: {std}")
    else:
        mean, std = compute_dataset_stats(
            os.path.join(data_dir, "train_lenses"),
            os.path.join(data_dir, "train_nonlenses")
        )
        np.savez(cache_path, mean=np.array(mean), std=np.array(std))

    train_transform.transforms.append(transforms.Normalize(mean, std))
    val_transform.transforms.append(transforms.Normalize(mean, std))

    train_dataset = LensDataset(os.path.join(data_dir, "train_lenses"),os.path.join(data_dir, "train_nonlenses"), transform=train_transform)

    test_dataset = LensDataset(os.path.join(data_dir, "test_lenses"),os.path.join(data_dir, "test_nonlenses"), transform=val_transform)

    labels = [s[1] for s in train_dataset.data]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader   = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    



    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")


    return train_loader, test_loader, mean, std

def compute_dataset_stats(lens_dir, nonlens_dir):
    all_files = (
        [os.path.join(lens_dir, f) for f in os.listdir(lens_dir) if f.endswith('.npy')] +
        [os.path.join(nonlens_dir, f) for f in os.listdir(nonlens_dir) if f.endswith('.npy')]
    )
    channel_sum  = np.zeros(3)
    channel_sum2 = np.zeros(3)
    n = 0
    for f in all_files:
        img = np.load(f).astype(np.float32)
        for c in range(3):
            mn, mx = img[c].min(), img[c].max()
            if mx > mn:
                img[c] = (img[c] - mn) / (mx - mn)
        channel_sum  += img.mean(axis=(1,2))
        channel_sum2 += (img**2).mean(axis=(1,2))
        n += 1
    mean = channel_sum / n
    std  = np.sqrt(channel_sum2/n - mean**2)
    print(f"Mean: {mean}, Std: {std}")
    return mean.tolist(), std.tolist()

def evaluate_roc_auc(loader, model, device="cuda", save_plot=True):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x      = x.to(device)
            logits = model(x).squeeze(1)          # (B,1) -> (B,)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Lens Finding")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plt.savefig("roc_curve.png", dpi=150)
        print("ROC curve saved.")

    plt.close()
    print(f"\n=== AUC Score: {auc:.4f} ===")
    model.train()
    return {"binary_auc": auc, "macro_avg": auc}