import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from dataset import LensingDataset
from torch.utils.data import DataLoader, random_split


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


def split_dataset(dataset, test_split=0.1, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size], generator=generator)


def get_loaders(data_dir, train_transform, val_transform, batch_size, num_workers, pin_memory):
    full_dataset = LensingDataset(data_dir, transform=None)

    train_dataset, test_dataset = split_dataset(full_dataset, test_split=0.1)

    train_dataset.dataset.transform = train_transform

    train_subset = _TransformSubset(full_dataset, train_dataset.indices, train_transform)
    val_subset   = _TransformSubset(full_dataset, test_dataset.indices,  val_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train samples: {len(train_subset)} | Val samples: {len(val_subset)}")
    return train_loader, val_loader


class _TransformSubset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.dataset.images[self.indices[idx]]
        label    = self.dataset.labels[self.indices[idx]]

        import numpy as np
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


def evaluate_roc_auc(loader, model, device="cuda", class_names=["no", "sphere", "vort"], save_plot=True):

    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   
    all_labels = np.concatenate(all_labels, axis=0)  

    num_classes = all_probs.shape[1]
    labels_onehot = np.eye(num_classes)[all_labels]   

    auc_scores = {}
    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], all_probs[:, i])
        auc = roc_auc_score(labels_onehot[:, i], all_probs[:, i])
        auc_scores[class_name] = auc
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.4f})")

    macro_auc = np.mean(list(auc_scores.values()))
    auc_scores["macro_avg"] = macro_auc

    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - DeepLense Multi-Class Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_plot:
        plt.savefig("roc_curves.png", dpi=150)
        print("ROC curve saved to roc_curves.png")

    plt.show()

    print("\n=== AUC Scores ===")
    for k, v in auc_scores.items():
        print(f"  {k}: {v:.4f}")

    model.train()
    return auc_scores
