import os
import copy
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\maher\Downloads\PressureUlc\dataset"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 15
NUM_CLASSES = 4
SEED = 42

SAVE_PATH = "best_pressure_ulcer_cnn.pth"

DEVICE = "cpu"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    # Makes results reproducible across runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Kept augmentation conservative for medical images
# Pressure ulcer staging depends on subtle color and tissue appearance
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.0,
        hue=0.0
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET SUBSET WRAPPER
# ============================================================

class IndexedSubset(Dataset):
    """
    Wrapper around ImageFolder so I can manually control
    which image indices belong to train/val/test.
    """

    def __init__(self, base_ds, indices, transform=None):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_i = self.indices[i]

        img_path, label = self.base_ds.samples[real_i]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# LOAD DATASET
# ============================================================

full_ds = datasets.ImageFolder(DATA_DIR)
class_names = full_ds.classes

print("Classes found:", class_names)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, got {len(class_names)}"
    )

labels = np.array(full_ds.targets)


# ============================================================
# TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# ============================================================

train_ids, tmp_ids = train_test_split(
    np.arange(len(labels)),
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

tmp_labels = labels[tmp_ids]

val_ids, test_ids = train_test_split(
    tmp_ids,
    test_size=0.50,
    random_state=SEED,
    stratify=tmp_labels
)

# Print test set files
print("\nTest set files:")
for idx in test_ids:
    print(f"  {full_ds.samples[idx][0]}")
print()


train_ds = IndexedSubset(full_ds, train_ids, transform=train_tfms)
val_ds   = IndexedSubset(full_ds, val_ids, transform=eval_tfms)
test_ds  = IndexedSubset(full_ds, test_ids, transform=eval_tfms)



# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

train_labels = labels[train_ids]
label_counts = Counter(train_labels)

print("Training set distribution:", label_counts)

counts = np.array(
    [label_counts[i] for i in range(NUM_CLASSES)],
    dtype=np.float32
)

# Smaller classes get larger loss weights
weights = counts.sum() / (NUM_CLASSES * counts)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

print("Class weights:", class_weights)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # Reduce spatial size while keeping strongest activations
            nn.MaxPool2d(2),

            # Helps regularize a bit
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.layers(x)


class PressureUlcerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = nn.Sequential(
            ConvBlock(3, 32, drop=0.10),
            ConvBlock(32, 64, drop=0.15),
            ConvBlock(64, 128, drop=0.20),
            ConvBlock(128, 256, drop=0.25)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
        return x


model = PressureUlcerCNN(NUM_CLASSES).to(DEVICE)

print(model)


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


# ============================================================
# TRAIN / EVAL HELPER
# ============================================================

def run_epoch(model, loader, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    preds_all = []
    labels_all = []

    for imgs, lbls in loader:

        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(imgs)
            loss = loss_fn(logits, lbls)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(labels_all, preds_all)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_all,
        preds_all,
        average="macro",
        zero_division=0
    )

    return avg_loss, acc, prec, rec, f1


# ============================================================
# TRAINING LOOP
# ============================================================

best_f1 = -1
best_weights = None
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):

    train_metrics = run_epoch(model, train_loader, training=True)
    val_metrics   = run_epoch(model, val_loader, training=False)

    train_loss, train_acc, _, _, train_f1 = train_metrics
    val_loss, val_acc, _, _, val_f1 = val_metrics

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
        f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_weights = copy.deepcopy(model.state_dict())

        torch.save(best_weights, SAVE_PATH)

        epochs_without_improvement = 0

        print(f"Saved new best model -> {SAVE_PATH}")

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(SAVE_PATH, map_location=DEVICE)
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_loss, test_acc, test_prec, test_rec, test_f1 = run_epoch(
    model,
    test_loader,
    training=False
)

print("\n=== TEST RESULTS ===")
print(f"Loss:      {test_loss:.4f}")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1 Score:  {test_f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

model.eval()

all_preds = []
all_truth = []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(DEVICE)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_truth.extend(lbls.numpy())

cm = confusion_matrix(all_truth, all_preds)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# LABEL MAP
# ============================================================

print("\nClass Mapping:")
for idx, name in enumerate(class_names):
    print(f"{idx} -> {name}")