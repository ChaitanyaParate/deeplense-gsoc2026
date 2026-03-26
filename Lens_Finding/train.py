from torchvision import transforms
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import model as mdel
from dataset import LensDataset
from torch.utils.data import DataLoader
from utils import get_loaders, save_checkpoint, load_checkpoint, evaluate_roc_auc
from loss import FocalLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt


LEARNING_RATE   = 1e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
WARMUP_EPOCHS   = 3      
NUM_WORKERS     = 2
PIN_MEMORY      = True
DATA_DIR       = "/mnt/newvolume/Programming/Python/Deep_Learning/deeplense-gsoc2026/Lens_Finding/lens-finding-test"

train_losses = []
val_auc_history = []

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
])

test_tf = transforms.Compose([

])


def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0

    for data, targets in loop:
        data    = data.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets.unsqueeze(1).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg = total_loss / len(loader)
    train_losses.append(avg)
    return avg

def main():
    print(f"Using device: {DEVICE}")

    train_loader, test_loader, mean, std = get_loaders(
        DATA_DIR, train_tf, test_tf,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    model = mdel.to(DEVICE)

    loss_fn   = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    #scaler    = torch.amp.GradScaler()

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn)
        print(f"  Avg Train Loss: {avg_loss:.4f}")

        auc_scores = evaluate_roc_auc(test_loader, model, device=DEVICE, save_plot=False)
        macro_auc  = auc_scores["macro_avg"]
        val_auc_history.append(macro_auc)

        print(f"  Macro AUC: {macro_auc:.4f}")

        if macro_auc > best_auc:
            best_auc = macro_auc
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "epoch":      epoch,
                "best_auc":   best_auc,
            })
            print(f"  Best model saved (AUC = {best_auc:.4f})")

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

    print("\n=== Final Evaluation on Validation Set ===")
    evaluate_roc_auc(test_loader, model, device=DEVICE, save_plot=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_auc_history)
    plt.title("Validation Macro AUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")


if __name__ == "__main__":
    main()
