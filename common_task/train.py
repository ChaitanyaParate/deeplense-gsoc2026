import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import build_vit, freeze_backbone, unfreeze_all
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    evaluate_roc_auc,
)

LEARNING_RATE   = 1e-4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64
NUM_EPOCHS      = 30
WARMUP_EPOCHS   = 3      
NUM_WORKERS     = 2
IMAGE_HEIGHT    = 224
IMAGE_WIDTH     = 224
PIN_MEMORY      = True
TRAIN_DIR       = "/mnt/newvolume/Programming/Python/Deep_Learning/deeplense-gsoc2026/common_task/dataset/dataset/train"
NUM_CLASSES     = 3

train_losses = []
val_auc_history = []

train_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=1.0),
    ToTensorV2(),
])


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0

    for data, targets in loop:
        data    = data.to(DEVICE)
        targets = targets.long().to(DEVICE)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    return avg_loss

def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader = get_loaders(
        TRAIN_DIR, train_transform, val_transform,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    model = build_vit(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
    scaler    = torch.amp.GradScaler()

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        if epoch < WARMUP_EPOCHS:
            freeze_backbone(model)
            print("  [Warmup] Backbone frozen - training head only")
        elif epoch == WARMUP_EPOCHS:
            unfreeze_all(model)
            print("  [Unfreeze] All layers now trainable")

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"  Avg Train Loss: {avg_loss:.4f}")

        auc_scores = evaluate_roc_auc(val_loader, model, device=DEVICE, save_plot=False)
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
    evaluate_roc_auc(val_loader, model, device=DEVICE, save_plot=True)

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
