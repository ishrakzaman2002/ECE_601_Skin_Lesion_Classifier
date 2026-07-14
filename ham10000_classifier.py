"""
ECE 601 - Machine Learning for Engineers, Spring 2026
Course Project - Phase 2
HAM10000 Skin Lesion Multi-Class Classifier
Generated via AI coding assistant using Phase 1 prompt.

=======================================================
COLAB SETUP INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. Create a new notebook
3. Go to Runtime > Change runtime type > Select T4 GPU
4. In the first cell, run:
       from google.colab import drive
       drive.mount('/content/drive')
       !pip install -q seaborn
5. Upload this script to Colab and run it, or paste
   the contents into a code cell.

DATASET SETUP:
- Download HAM10000 from Kaggle:
  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Upload the contents to your Google Drive under:
    My Drive/HAM10000/
  so the folder contains:
    HAM10000_metadata.csv
    HAM10000_images_part1/   (folder of .jpg images)
    HAM10000_images_part2/   (folder of .jpg images)
=======================================================
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 0. Reproducibility
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
# Colab path (Google Drive):
DATA_DIR  = r"C:\pracona\601\HAM10000"

CSV_PATH  = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMG_DIR_1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 1e-4
NUM_WORKERS = 0       # must be 0 on Windows to avoid multiprocessing errors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 2. Label mapping
# ─────────────────────────────────────────────
LABEL_MAP = {
    "mel":   0,
    "nv":    1,
    "bcc":   2,
    "akiec": 3,
    "bkl":   4,
    "df":    5,
    "vasc":  6,
}
CLASS_NAMES = list(LABEL_MAP.keys())

# ─────────────────────────────────────────────
# 3. Build image path lookup & dataframe
# ─────────────────────────────────────────────
def find_image_path(image_id):
    for img_dir in [IMG_DIR_1, IMG_DIR_2]:
        path = os.path.join(img_dir, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

df = pd.read_csv(CSV_PATH)
df["label"] = df["dx"].map(LABEL_MAP)
df["path"]  = df["image_id"].apply(find_image_path)
df = df.dropna(subset=["path"])

print(f"\nDataset size: {len(df)} images")
print("\nClass distribution:")
print(df["dx"].value_counts())

# ─────────────────────────────────────────────
# 4. Train / Val / Test split  (70 / 15 / 15)
# ─────────────────────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
)

print(f"\nSplit — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ─────────────────────────────────────────────
# 5. Transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 6. Dataset class
# ─────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = SkinLesionDataset(train_df, transform=train_transform)
val_dataset   = SkinLesionDataset(val_df,   transform=val_test_transform)
test_dataset  = SkinLesionDataset(test_df,  transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)

# ─────────────────────────────────────────────
# 7. Model — ResNet-18 with fine-tuned classifier head
# ─────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace final FC layer to output 7 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(LABEL_MAP))
model = model.to(DEVICE)

print(f"\nModel: ResNet-18 | Output classes: {len(LABEL_MAP)}")

# ─────────────────────────────────────────────
# 8. Loss, optimizer, scheduler
# ─────────────────────────────────────────────
# Inverse-frequency class weights to address class imbalance
class_counts = df["label"].value_counts().sort_index().values
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(LABEL_MAP)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)
# Decay LR by 0.1 every 7 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─────────────────────────────────────────────
# 9. Training loop
# ─────────────────────────────────────────────
def run_epoch(loader, model, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_acc = 0.0

print("\n--- Training ---")
for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer, train=True)
    va_loss, va_acc = run_epoch(val_loader,   model, criterion, train=False)
    scheduler.step()

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    train_accs.append(tr_acc)
    val_accs.append(va_acc)

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
          f"Val Loss: {va_loss:.4f}  Val Acc: {va_acc:.4f}")

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

# ─────────────────────────────────────────────
# 10. Plot training curves
# ─────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, train_losses, label="Train Loss")
axes[0].plot(epochs_range, val_losses,   label="Val Loss")
axes[0].set_title("Loss vs. Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs_range, train_accs, label="Train Accuracy")
axes[1].plot(epochs_range, val_accs,   label="Val Accuracy")
axes[1].set_title("Accuracy vs. Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")

# ─────────────────────────────────────────────
# 11. Test set evaluation
# ─────────────────────────────────────────────
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
macro_f1 = f1_score(all_labels, all_preds, average="macro")

print("\n--- Test Set Results ---")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print("\nPer-Class Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 12. Confusion matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")
