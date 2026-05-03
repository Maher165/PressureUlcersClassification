import os
import cv2
import numpy as np
from deepskin import wound_segmentation

input_folder = "C:\\Users\\maher\\Downloads\\PressureUlc\\UlcersPressure\\Stage_IV"
output_folder = "C:\\Users\\maher\\Downloads\\PressureUlc\\UlcersPressure\\Stage_IV_Masks"
overlay_folder = "dataset/overlay_results_IV"   # Overlay on original image

os.makedirs(output_folder, exist_ok=True)
os.makedirs(overlay_folder, exist_ok=True)

# =========================
# PROCESS IMAGES
# =========================
for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(input_folder, file)

        # Load original image
        bgr = cv2.imread(img_path)

        if bgr is None:
            print(f"Could not load {file}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # =========================
        # RUN DEEPSKIN
        # =========================
        mask = wound_segmentation(img=rgb)

        # =========================
        # HANDLE MULTI-CLASS / PROBABILITY OUTPUT
        # =========================
        if len(mask.shape) == 3:
            # If probabilities/logits
            mask = np.argmax(mask, axis=-1)

        print(f"{file} -> Unique values: {np.unique(mask)}")

        # =========================
        # TRY TO DETECT WOUND CLASS
        # Usually wound is highest class
        # =========================
        wound_class = np.max(mask)

        # Binary wound mask
        binary_wound = (mask == wound_class).astype(np.uint8) * 255

        # =========================
        # ENLARGE SMALL DETAILS
        # =========================
        kernel = np.ones((3, 3), np.uint8)
        binary_wound = cv2.morphologyEx(binary_wound, cv2.MORPH_CLOSE, kernel)
        binary_wound = cv2.dilate(binary_wound, kernel, iterations=1)

        # =========================
        # SAVE HIGH-CONTRAST MASK
        # White wound / Black background
        # =========================
        mask_save_path = os.path.join(output_folder, file)
        cv2.imwrite(mask_save_path, binary_wound)

        # =========================
        # CREATE RED OVERLAY
        # =========================
        overlay = bgr.copy()

        # Red wound area
        overlay[binary_wound == 255] = [0, 0, 255]

        # Blend with original
        blended = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

        # Add contour for better visibility
        contours, _ = cv2.findContours(binary_wound, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

        overlay_save_path = os.path.join(overlay_folder, file)
        cv2.imwrite(overlay_save_path, blended)

        print(f"Saved mask: {mask_save_path}")
        print(f"Saved overlay: {overlay_save_path}")

print("Done. Check both predicted_masks and overlay_results folders.")