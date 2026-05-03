# -----------------------
# Test one image after training (show original + predicted mask)
# -----------------------
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_wound_model.pth"
TEST_IMAGE = r"C:\Users\maher\Downloads\PressureUlc\UlcersPressure\Stage_II\Stage_II_011.png"


# -----------------------
# Helper: read image
# -----------------------
def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# -----------------------
# Load model
# -----------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# -----------------------
# Prediction function
# -----------------------
@torch.no_grad()
def predict_mask(model, image_path):
    image = read_image(image_path)
    h, w = image.shape[:2]

    tfm = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    sample = tfm(image=image)
    x = sample["image"].unsqueeze(0).to(DEVICE)

    logits = model(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


# -----------------------
# Run inference
# -----------------------
pred_mask = predict_mask(model, TEST_IMAGE)
original = read_image(TEST_IMAGE)


# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
#plt.imshow(original)
#plt.title("Original Image")
#plt.axis("off")

#plt.subplot(1,2,2)
#plt.imshow(pred_mask, cmap="gray")
#plt.title("Predicted Mask")
#plt.axis("off")

#plt.show()


plt.imshow(original)
plt.imshow(pred_mask, cmap="jet", alpha=0.4)
plt.title("Wound Overlay")
plt.axis("off")
plt.show()
# Optional: save predicted mask
#cv2.imwrite("predicted_mask.png", pred_mask)
#print("Saved as predicted_mask.png")


'''Epoch 01/40 | train_loss=0.6966 | val_loss=0.6457 | val_dice=0.4981                                 
Saved best model to best_wound_model.pth
Epoch 02/40 | train_loss=0.6169 | val_loss=0.5880 | val_dice=0.5595                                 
Saved best model to best_wound_model.pth
Epoch 03/40 | train_loss=0.5775 | val_loss=0.5404 | val_dice=0.6104                                 
Saved best model to best_wound_model.pth
Epoch 04/40 | train_loss=0.5381 | val_loss=0.4951 | val_dice=0.6475                                 
Saved best model to best_wound_model.pth
Epoch 05/40 | train_loss=0.5082 | val_loss=0.4779 | val_dice=0.6665                                 
Saved best model to best_wound_model.pth
Epoch 06/40 | train_loss=0.4906 | val_loss=0.4880 | val_dice=0.6352                                 
Epoch 07/40 | train_loss=0.5065 | val_loss=0.5382 | val_dice=0.4026                                 
Epoch 08/40 | train_loss=0.4687 | val_loss=0.5232 | val_dice=0.4651                                 
Epoch 09/40 | train_loss=0.4363 | val_loss=0.4178 | val_dice=0.7146                                 
Saved best model to best_wound_model.pth
Epoch 10/40 | train_loss=0.4453 | val_loss=0.4687 | val_dice=0.6434                                 
Epoch 11/40 | train_loss=0.4135 | val_loss=0.4849 | val_dice=0.5739                                 
Epoch 12/40 | train_loss=0.4590 | val_loss=0.4522 | val_dice=0.6345                                 
Epoch 13/40 | train_loss=0.4399 | val_loss=0.4187 | val_dice=0.6543                                 
Epoch 14/40 | train_loss=0.4086 | val_loss=0.4184 | val_dice=0.6645                                 
Epoch 15/40 | train_loss=0.4343 | val_loss=0.3844 | val_dice=0.7508                                 
Saved best model to best_wound_model.pth
Epoch 16/40 | train_loss=0.3917 | val_loss=0.4136 | val_dice=0.7136                                 
Epoch 17/40 | train_loss=0.4214 | val_loss=0.3898 | val_dice=0.7251                                 
Epoch 18/40 | train_loss=0.3973 | val_loss=0.4562 | val_dice=0.5858                                 
Epoch 19/40 | train_loss=0.3639 | val_loss=0.4084 | val_dice=0.7012                                 
Epoch 20/40 | train_loss=0.3672 | val_loss=0.4030 | val_dice=0.6995                                 
Epoch 21/40 | train_loss=0.3849 | val_loss=0.3978 | val_dice=0.7158                                 
Epoch 22/40 | train_loss=0.3847 | val_loss=0.3797 | val_dice=0.7082                                 
Epoch 23/40 | train_loss=0.3559 | val_loss=0.3795 | val_dice=0.7203                                 
Epoch 24/40 | train_loss=0.3944 | val_loss=0.3561 | val_dice=0.7494                                 
Epoch 25/40 | train_loss=0.3786 | val_loss=0.3770 | val_dice=0.7257                                 
Epoch 26/40 | train_loss=0.3334 | val_loss=0.3773 | val_dice=0.7280                                 
Epoch 27/40 | train_loss=0.3335 | val_loss=0.3776 | val_dice=0.7319                                 
Epoch 28/40 | train_loss=0.3630 | val_loss=0.3745 | val_dice=0.7164                                 
Epoch 29/40 | train_loss=0.3765 | val_loss=0.3693 | val_dice=0.7084                                 
Epoch 30/40 | train_loss=0.3694 | val_loss=0.3653 | val_dice=0.7237                                 
Epoch 31/40 | train_loss=0.3388 | val_loss=0.4080 | val_dice=0.6488                                 
Epoch 32/40 | train_loss=0.3409 | val_loss=0.3753 | val_dice=0.7050                                 
Epoch 33/40 | train_loss=0.3751 | val_loss=0.3546 | val_dice=0.7205                                 
Epoch 34/40 | train_loss=0.3058 | val_loss=0.3580 | val_dice=0.7153                                 
Epoch 35/40 | train_loss=0.3647 | val_loss=0.3669 | val_dice=0.7140                                 
Epoch 36/40 | train_loss=0.3952 | val_loss=0.3693 | val_dice=0.7221                                 
Epoch 37/40 | train_loss=0.3447 | val_loss=0.3576 | val_dice=0.7313                                 
Epoch 38/40 | train_loss=0.3647 | val_loss=0.3530 | val_dice=0.7472                                 
Epoch 39/40 | train_loss=0.3529 | val_loss=0.3369 | val_dice=0.7452                                 
Epoch 40/40 | train_loss=0.3049 | val_loss=0.3820 | val_dice=0.6896                                 
Training finished.'''