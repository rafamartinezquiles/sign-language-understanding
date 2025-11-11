"""
Gesture Collage Generator
-------------------------
This script creates a visual collage of randomly sampled gesture images.
Each row of the collage contains a fixed number of gesture examples, allowing
a quick overview of all gesture categories.

Workflow:
1. The "gestures" directory should contain one subfolder per gesture (named numerically).
2. Each folder must contain gesture images (e.g., 1.jpg, 2.jpg, ...).
3. The script will:
   - Randomly select one image from each gesture folder.
   - Arrange 5 gesture categories per row in a collage grid.
   - Display and save the final combined image as 'full_img.jpg'.

Features:
- Automatically adapts to number of gesture folders
- Skips missing images safely
- Uses consistent random sampling
- Works with any number of gestures and images per folder

Dependencies:
    OpenCV, NumPy, OS, Random
"""

import cv2
import os
import random
import numpy as np


# ===============================================================
# Configuration
# ===============================================================

GESTURE_FOLDER = "gestures"
GESTURES_PER_ROW = 5
IMAGES_PER_GESTURE = 200  # expected number of gesture images per folder


# ===============================================================
# Utility Functions
# ===============================================================

def get_image_size():
    """Find the first valid gesture image in the dataset and return its size."""
    if not os.path.exists(GESTURE_FOLDER):
        raise FileNotFoundError(f"'{GESTURE_FOLDER}' directory not found.")

    # Get gesture folders (skip non-digits safely)
    gesture_folders = sorted(
        [g for g in os.listdir(GESTURE_FOLDER) if os.path.isdir(os.path.join(GESTURE_FOLDER, g))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    if not gesture_folders:
        raise FileNotFoundError("No gesture folders found in 'gestures/'.")

    # Loop through all gesture folders and look for any .jpg image
    for folder in gesture_folders:
        folder_path = os.path.join(GESTURE_FOLDER, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img.shape

    raise FileNotFoundError("No valid .jpg images found in any gesture folder.")


def safe_read_image(path, image_size):
    """Safely read an image, or return a blank placeholder if not found."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros(image_size, dtype=np.uint8)
    return img


# ===============================================================
# Collage Builder
# ===============================================================

def build_gesture_collage():
    """Build a visual grid collage of randomly sampled gesture images."""
    if not os.path.exists(GESTURE_FOLDER):
        print(f"Error: '{GESTURE_FOLDER}' directory not found.")
        return

    gestures = [g for g in os.listdir(GESTURE_FOLDER)
                if os.path.isdir(os.path.join(GESTURE_FOLDER, g))]
    if not gestures:
        print("No gesture subfolders found inside 'gestures/'.")
        return

    # Sort numerically (1, 2, 3, ...)
    gestures.sort(key=lambda x: int(x) if x.isdigit() else x)
    image_y, image_x = get_image_size()

    total_gestures = len(gestures)
    rows = total_gestures // GESTURES_PER_ROW + (1 if total_gestures % GESTURES_PER_ROW != 0 else 0)

    print(f"Creating collage for {total_gestures} gestures ({rows} rows of up to {GESTURES_PER_ROW})...")

    full_img = None
    begin_index = 0
    end_index = GESTURES_PER_ROW

    for i in range(rows):
        col_img = None

        for j in range(begin_index, min(end_index, total_gestures)):
            gesture_id = gestures[j]
            gesture_path = os.path.join(GESTURE_FOLDER, gesture_id)
            image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(".jpg")]

            if not image_files:
                print(f"Warning: No images found in {gesture_path}. Using blank placeholder.")
                img = np.zeros((image_y, image_x), dtype=np.uint8)
            else:
                img_index = random.choice(image_files)
                img_path = os.path.join(gesture_path, img_index)
                img = safe_read_image(img_path, (image_y, image_x))

            if col_img is None:
                col_img = img
            else:
                col_img = np.hstack((col_img, img))

        begin_index += GESTURES_PER_ROW
        end_index += GESTURES_PER_ROW

        if full_img is None:
            full_img = col_img
        else:
            full_img = np.vstack((full_img, col_img))

    if full_img is not None:
        cv2.imshow("Gesture Collage", full_img)
        cv2.imwrite("full_img.jpg", full_img)
        print("Collage saved as 'full_img.jpg'.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid images available to build collage.")


# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    build_gesture_collage()