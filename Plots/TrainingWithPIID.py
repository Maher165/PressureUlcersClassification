import re
import matplotlib.pyplot as plt

log_text = """Epoch 01/40 | Train Loss 1.1884 Acc 0.4351 F1 0.4223 | Val Loss 1.5782 Acc 0.2378 F1 0.1484
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 02/40 | Train Loss 1.0735 Acc 0.4849 F1 0.4780 | Val Loss 1.0428 Acc 0.4695 F1 0.4604
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 03/40 | Train Loss 1.0323 Acc 0.5636 F1 0.5587 | Val Loss 0.9722 Acc 0.4817 F1 0.4329
Epoch 04/40 | Train Loss 0.9508 Acc 0.5872 F1 0.5893 | Val Loss 1.0234 Acc 0.4878 F1 0.4635
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 05/40 | Train Loss 0.9405 Acc 0.5898 F1 0.5906 | Val Loss 1.0045 Acc 0.5122 F1 0.4766
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 06/40 | Train Loss 0.9406 Acc 0.6003 F1 0.5983 | Val Loss 0.9022 Acc 0.6159 F1 0.6123
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 07/40 | Train Loss 0.9094 Acc 0.6107 F1 0.6089 | Val Loss 0.9807 Acc 0.5366 F1 0.5312
Epoch 08/40 | Train Loss 0.8962 Acc 0.6121 F1 0.6140 | Val Loss 0.9538 Acc 0.6463 F1 0.6492
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 09/40 | Train Loss 0.9059 Acc 0.6278 F1 0.6293 | Val Loss 0.8818 Acc 0.5732 F1 0.5534
Epoch 10/40 | Train Loss 0.8708 Acc 0.6606 F1 0.6592 | Val Loss 0.9527 Acc 0.5549 F1 0.5496
Epoch 11/40 | Train Loss 0.8928 Acc 0.6330 F1 0.6350 | Val Loss 0.8911 Acc 0.6402 F1 0.6397
Epoch 12/40 | Train Loss 0.8799 Acc 0.6239 F1 0.6265 | Val Loss 1.0271 Acc 0.5244 F1 0.5159
Epoch 13/40 | Train Loss 0.8301 Acc 0.6658 F1 0.6669 | Val Loss 0.9294 Acc 0.6037 F1 0.5853
Epoch 14/40 | Train Loss 0.7967 Acc 0.6710 F1 0.6742 | Val Loss 0.9034 Acc 0.6280 F1 0.6105
Epoch 15/40 | Train Loss 0.7628 Acc 0.7248 F1 0.7270 | Val Loss 0.9018 Acc 0.5976 F1 0.5811
Epoch 16/40 | Train Loss 0.7977 Acc 0.7038 F1 0.7058 | Val Loss 0.8096 Acc 0.6829 F1 0.6803
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 17/40 | Train Loss 0.8072 Acc 0.7169 F1 0.7207 | Val Loss 0.8278 Acc 0.6951 F1 0.6975
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 18/40 | Train Loss 0.7448 Acc 0.7261 F1 0.7283 | Val Loss 0.9156 Acc 0.6402 F1 0.6440
Epoch 19/40 | Train Loss 0.7722 Acc 0.7090 F1 0.7104 | Val Loss 0.8631 Acc 0.6768 F1 0.6761
Epoch 20/40 | Train Loss 0.8070 Acc 0.6881 F1 0.6919 | Val Loss 0.9161 Acc 0.5976 F1 0.5947
Epoch 21/40 | Train Loss 0.7122 Acc 0.7497 F1 0.7522 | Val Loss 0.9249 Acc 0.6585 F1 0.6585
Epoch 22/40 | Train Loss 0.7037 Acc 0.7628 F1 0.7656 | Val Loss 0.8033 Acc 0.6585 F1 0.6612
Epoch 23/40 | Train Loss 0.7190 Acc 0.7431 F1 0.7460 | Val Loss 0.7964 Acc 0.6890 F1 0.6917
Epoch 24/40 | Train Loss 0.7152 Acc 0.7523 F1 0.7541 | Val Loss 0.8327 Acc 0.6951 F1 0.6992
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 25/40 | Train Loss 0.7273 Acc 0.7379 F1 0.7402 | Val Loss 0.8031 Acc 0.6951 F1 0.6996
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 26/40 | Train Loss 0.7341 Acc 0.7326 F1 0.7336 | Val Loss 0.8180 Acc 0.6524 F1 0.6550
Epoch 27/40 | Train Loss 0.6907 Acc 0.7562 F1 0.7591 | Val Loss 0.8676 Acc 0.6829 F1 0.6850
Epoch 28/40 | Train Loss 0.7028 Acc 0.7418 F1 0.7429 | Val Loss 0.8112 Acc 0.6646 F1 0.6667
Epoch 29/40 | Train Loss 0.6856 Acc 0.7628 F1 0.7657 | Val Loss 0.7928 Acc 0.7012 F1 0.7054
Saved new best model -> best_pressure_ulcer_cnn.pth
Epoch 30/40 | Train Loss 0.6561 Acc 0.7890 F1 0.7914 | Val Loss 0.8244 Acc 0.6768 F1 0.6794
Epoch 31/40 | Train Loss 0.6680 Acc 0.7693 F1 0.7709 | Val Loss 0.8216 Acc 0.6829 F1 0.6872
Epoch 32/40 | Train Loss 0.6807 Acc 0.7785 F1 0.7812 | Val Loss 0.8367 Acc 0.6585 F1 0.6624
Epoch 33/40 | Train Loss 0.6608 Acc 0.7693 F1 0.7719 | Val Loss 0.8111 Acc 0.6829 F1 0.6855
Epoch 34/40 | Train Loss 0.6356 Acc 0.8021 F1 0.8044 | Val Loss 0.8129 Acc 0.6829 F1 0.6865
Epoch 35/40 | Train Loss 0.6472 Acc 0.7890 F1 0.7912 | Val Loss 0.8477 Acc 0.6585 F1 0.6609
Epoch 36/40 | Train Loss 0.6509 Acc 0.7903 F1 0.7923 | Val Loss 0.8446 Acc 0.6524 F1 0.6550
Epoch 37/40 | Train Loss 0.6688 Acc 0.7680 F1 0.7703 | Val Loss 0.8456 Acc 0.6524 F1 0.6537
Epoch 38/40 | Train Loss 0.6589 Acc 0.7772 F1 0.7796 | Val Loss 0.8601 Acc 0.6646 F1 0.6661
Epoch 39/40 | Train Loss 0.6363 Acc 0.7903 F1 0.7921 | Val Loss 0.8256 Acc 0.6585 F1 0.6611
Epoch 40/40 | Train Loss 0.6594 Acc 0.7837 F1 0.7866 | Val Loss 0.8256 Acc 0.6768 F1 0.6805

=== TEST RESULTS ===
Loss:      0.8029
Accuracy:  0.6646
Precision: 0.6656
Recall:    0.6703
F1 Score:  0.6606

Confusion Matrix:
[[30  3  1  1]
 [12 33  1  1]
 [ 0  4 28  9]
 [ 0  3 20 18]]

Class Mapping:
0 -> 1
1 -> 2
2 -> 3
3 -> 4"""

# Regex to extract values
pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss ([\d.]+) Acc ([\d.]+) F1 ([\d.]+)\s+\|\s+Val Loss ([\d.]+) Acc ([\d.]+) F1 ([\d.]+)"
)

epochs = []
train_loss, train_acc, train_f1 = [], [], []
val_loss, val_acc, val_f1 = [], [], []

for match in pattern.finditer(log_text):
    epochs.append(int(match.group(1)))
    train_loss.append(float(match.group(2)))
    train_acc.append(float(match.group(3)))
    train_f1.append(float(match.group(4)))
    val_loss.append(float(match.group(5)))
    val_acc.append(float(match.group(6)))
    val_f1.append(float(match.group(7)))

# Plot
plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()

# F1 Score
plt.subplot(1, 3, 3)
plt.plot(epochs, train_f1, label='Train F1')
plt.plot(epochs, val_f1, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Epoch')
plt.legend()

plt.tight_layout()
plt.show()