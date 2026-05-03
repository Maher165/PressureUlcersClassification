from pathlib import Path
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# -----------------------
# Config
# -----------------------
IMAGE_DIR = Path(r"C:\Users\maher\Downloads\PressureUlc\UlcersPressure\Stage_I")
MASK_DIR = Path(r"C:\Users\maher\Downloads\PressureUlc\PressureUlcersMasks\PressureUlcersMasks\Stage 1")

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-4
SEED = 42
THRESHOLD = 0.5
INVERT_MASKS = False   # set True if your masks are the opposite of what you described

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Reproducibility
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# File pairing
# -----------------------
def get_pairs(image_dir, mask_dir):
    # Image files are PNG
    image_exts = [".png"]

    # Mask files are TIFF
    mask_exts = [".tiff", ".tif"]

    pairs = []

    for img_path in sorted(image_dir.iterdir()):
        # Only read PNG images
        if img_path.suffix.lower() not in image_exts:
            continue

        mask_path = None

        # Look for matching TIFF mask with same filename
        # Example:
        # images/wound_001.png -> masks/wound_001.tiff
        for ext in mask_exts:
            candidate = mask_dir / f"{img_path.stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"No corresponding TIFF mask found for image: {img_path.name}"
            )

        pairs.append((img_path, mask_path))

    if len(pairs) == 0:
        raise FileNotFoundError("No image-mask pairs found.")

    print(f"Found {len(pairs)} image-mask pairs.")
    return pairs


# -----------------------
# Loading
# -----------------------
def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")

    if INVERT_MASKS:
        mask = 255 - mask

    # Your description: wound = black, background = white
    # convert wound to 1, background to 0
    mask = (mask < 128).astype(np.float32)
    return mask


# -----------------------
# Dataset
# -----------------------
class WoundDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = read_image(img_path)
        mask = read_mask(mask_path)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        mask = mask.unsqueeze(0).float()  # [1, H, W]
        return image.float(), mask


# -----------------------
# Augmentations
# -----------------------
def train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# -----------------------
# Loss + metric
# -----------------------
def dice_score_from_probs(preds, targets, eps=1e-7):
    preds = preds.contiguous().view(preds.shape[0], -1)
    targets = targets.contiguous().view(targets.shape[0], -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (union + eps)
    return dice.mean()

class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return 1.0 - dice_score_from_probs(probs, targets)

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


# -----------------------
# Train / eval
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    dices = []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).float()

        losses.append(loss.item())
        dices.append(dice_score_from_probs(preds, masks).item())

    return float(np.mean(losses)), float(np.mean(dices))


# -----------------------
# Prediction
# -----------------------
@torch.no_grad()
def predict_mask(model, image_path, save_path=None):
    model.eval()
    image = read_image(image_path)
    h, w = image.shape[:2]

    tfm = val_transform()
    sample = tfm(image=image, mask=np.zeros((h, w), dtype=np.float32))
    x = sample["image"].unsqueeze(0).to(DEVICE)

    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob > THRESHOLD).astype(np.uint8) * 255

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    if save_path is not None:
        cv2.imwrite(str(save_path), pred)

    return pred


# -----------------------
# Main
# -----------------------
def main():
    seed_everything(SEED)

    pairs = get_pairs(IMAGE_DIR, MASK_DIR)
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=SEED)

    train_ds = WoundDataset(train_pairs, transform=train_transform())
    val_ds = WoundDataset(val_pairs, transform=val_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_dice = -1.0
    best_path = "best_wound_model.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "threshold": THRESHOLD,
                },
                best_path
            )
            print(f"Saved best model to {best_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()