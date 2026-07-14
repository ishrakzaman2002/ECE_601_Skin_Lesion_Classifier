"""
ECE 601 - Machine Learning for Engineers, Spring 2026
Course Project - Phase 5: Final Comparison

Compares two approaches on HAM10000:
    1. Our Method:     ResNet-50 + WeightedRandomSampler (best from Phase 3)
    2. Competitor:     ResNet-50 + Soft Attention (re-implemented from Khoi et al., 2022)

Results saved to: C:\pracona\601\phase5_results.txt
                  C:\pracona\601\phase5_results.csv
Plots saved to:   C:\pracona\601\phase5_plots\
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
SEED        = 42
DATA_DIR    = r"C:\pracona\601\HAM10000"
CSV_PATH    = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMG_DIR_1   = os.path.join(DATA_DIR, "HAM10000_images_part_1")
IMG_DIR_2   = os.path.join(DATA_DIR, "HAM10000_images_part_2")
OUT_DIR     = r"C:\pracona\601\phase5_plots"
LOG_PATH    = r"C:\pracona\601\phase5_results.txt"
CSV_OUT     = r"C:\pracona\601\phase5_results.csv"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 1e-4
NUM_WORKERS = 0

LABEL_MAP = {
    "mel": 0, "nv": 1, "bcc": 2,
    "akiec": 3, "bkl": 4, "df": 5, "vasc": 6,
}
CLASS_NAMES = list(LABEL_MAP.keys())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Logging
# ─────────────────────────────────────────────
# Clear log file at start
open(LOG_PATH, "w").close()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ─────────────────────────────────────────────
# 2. Reproducibility
# ─────────────────────────────────────────────
def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
def find_image_path(image_id):
    for img_dir in [IMG_DIR_1, IMG_DIR_2]:
        path = os.path.join(img_dir, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

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

# ─────────────────────────────────────────────
# 4. Transforms
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 5. Soft Attention Module (Khoi et al., 2022)
# ─────────────────────────────────────────────
class SoftAttention(nn.Module):
    """
    Soft Attention module as described in Khoi et al. (2022).
    Takes feature maps from the CNN backbone and produces
    an attention-weighted representation.
    The attention map is computed by a 1x1 convolution followed
    by a sigmoid activation, acting as a spatial gate over the
    feature maps. This allows the model to focus on diagnostically
    relevant regions of the lesion image.
    """
    def __init__(self, in_channels):
        super(SoftAttention, self).__init__()
        # 1x1 conv to produce a single-channel attention map
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        attn = self.attention_conv(x)       # (B, 1, H, W)
        attn = self.sigmoid(attn)           # normalize to [0, 1]
        x_attended = x * attn              # element-wise spatial weighting
        # Global average pool the attended features
        out = F.adaptive_avg_pool2d(x_attended, 1)  # (B, C, 1, 1)
        out = out.view(out.size(0), -1)             # (B, C)
        return out, attn


# ─────────────────────────────────────────────
# 6. Model Definitions
# ─────────────────────────────────────────────

class ResNet50Baseline(nn.Module):
    """
    Our method from Phase 3: ResNet-50 pretrained on ImageNet,
    fine-tuned end-to-end with a replaced classification head.
    """
    def __init__(self, num_classes):
        super(ResNet50Baseline, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)           # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 2048)
        return self.fc(x)


class ResNet50SoftAttention(nn.Module):
    """
    Competitor: ResNet-50 + Soft Attention (re-implemented from Khoi et al., 2022).
    The backbone feature maps (before global average pooling) are fed into
    the soft attention module, which produces a spatially-weighted representation
    that the classifier then uses.
    """
    def __init__(self, num_classes):
        super(ResNet50SoftAttention, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Feature extractor: everything up to (but not including) avgpool
        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
        # Soft attention over the 2048-channel feature maps
        self.soft_attention = SoftAttention(in_channels=2048)
        # Classifier head
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)          # (B, 2048, 7, 7)
        attended, attn_map = self.soft_attention(feats)  # (B, 2048)
        return self.fc(attended)

# ─────────────────────────────────────────────
# 7. Training helpers
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, train=True):
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
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc      = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    cm       = confusion_matrix(all_labels, all_preds)
    return acc, macro_f1, report, cm, all_preds, all_labels


def save_curves(train_losses, val_losses, train_accs, val_accs, name):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses,   label="Val Loss")
    axes[0].set_title(f"Loss — {name}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(epochs, train_accs, label="Train Acc")
    axes[1].plot(epochs, val_accs,   label="Val Acc")
    axes[1].set_title(f"Accuracy — {name}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved: {path}")


def save_confusion_matrix(cm, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_confusion.png")
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved: {path}")

# ─────────────────────────────────────────────
# 8. Main experiment runner
# ─────────────────────────────────────────────
def run_experiment(name, model, train_df, val_df, test_df, df_full):
    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {name}")
    log(f"{'='*60}")
    set_seed()

    # Class weights for loss
    class_counts = df_full["label"].value_counts().sort_index().values
    weights = torch.tensor(
        (1.0 / class_counts) / (1.0 / class_counts).sum() * len(LABEL_MAP),
        dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # WeightedRandomSampler for both experiments (our Phase 3 best setting)
    sample_weights = [1.0 / class_counts[label] for label in train_df["label"].values]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(SkinLesionDataset(train_df, train_transform),
                              batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(SkinLesionDataset(val_df,   val_test_transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(SkinLesionDataset(test_df,  val_test_transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_')}_best.pth")

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, train=False)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);   val_accs.append(va_acc)

        log(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {va_loss:.4f}  Val Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    log(f"  Best Val Acc: {best_val_acc:.4f}")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(best_path, weights_only=True))
    acc, macro_f1, report, cm, preds, labels = evaluate(model, test_loader)

    log(f"\n  Test Accuracy : {acc:.4f}")
    log(f"  Macro F1 Score: {macro_f1:.4f}")
    log(f"\n  Per-Class Report:\n{report}")

    # Save plots
    save_curves(train_losses, val_losses, train_accs, val_accs, name)
    save_confusion_matrix(cm, name)

    return {
        "Experiment":    name,
        "Test Accuracy": round(acc, 4),
        "Macro F1":      round(macro_f1, 4),
        "Per-Class Report": report,
        "Best Val Acc":  round(best_val_acc, 4),
    }

# ─────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log(f"Using device: {DEVICE}")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["dx"].map(LABEL_MAP)
    df["path"]  = df["image_id"].apply(find_image_path)
    df = df.dropna(subset=["path"])
    log(f"Dataset: {len(df)} images")
    log(f"Class distribution:\n{df['dx'].value_counts().to_string()}")

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
    log(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    results = []

    # Experiment 1: Our method — ResNet-50 baseline
    model_ours = ResNet50Baseline(num_classes=len(LABEL_MAP))
    r1 = run_experiment("Our Method ResNet-50", model_ours, train_df, val_df, test_df, df)
    results.append(r1)

    # Experiment 2: Competitor — ResNet-50 + Soft Attention
    model_attn = ResNet50SoftAttention(num_classes=len(LABEL_MAP))
    r2 = run_experiment("Competitor ResNet-50 Soft Attention", model_attn, train_df, val_df, test_df, df)
    results.append(r2)

    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(CSV_OUT, index=False)

    # Print final summary
    log("\n" + "="*60)
    log("FINAL COMPARISON SUMMARY")
    log("="*60)
    for r in results:
        log(f"\n{r['Experiment']}")
        log(f"  Test Accuracy : {r['Test Accuracy']}")
        log(f"  Macro F1      : {r['Macro F1']}")
    log(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results saved to: {CSV_OUT}")
    log(f"Plots saved to:   {OUT_DIR}")
    log(f"Full log saved to: {LOG_PATH}")
