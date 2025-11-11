"""
Gesture Image Flipper
---------------------
This script augments the gesture dataset by creating horizontally flipped
versions of each captured hand image. For every image in each gesture folder,
a mirrored version is saved with a new index number.

Workflow:
1. Ensure that the "gestures" directory exists and contains gesture subfolders.
2. Run this script.
3. Each original image will have a flipped duplicate added to the same folder.

Features:
- Automatically scans all gesture folders under "gestures/"
- Processes up to TOTAL_IMAGES per gesture (adjustable)
- Saves flipped copies with new sequential filenames
- Gracefully skips missing images or invalid files

Dependencies:
    OpenCV, OS
"""

import cv2
import os


# ===============================================================
# Global Parameters
# ===============================================================

GESTURES_FOLDER = "gestures"
TOTAL_IMAGES = 200  # number of original images per gesture


# ===============================================================
# Core Functionality
# ===============================================================

def flip_images():
    """Generate horizontally flipped copies of all gesture images."""
    if not os.path.exists(GESTURES_FOLDER):
        print(f"Error: '{GESTURES_FOLDER}' directory not found.")
        return

    gesture_folders = [f for f in os.listdir(GESTURES_FOLDER)
                       if os.path.isdir(os.path.join(GESTURES_FOLDER, f))]

    if not gesture_folders:
        print("No gesture folders found inside 'gestures/'.")
        return

    print(f"Processing {len(gesture_folders)} gesture folders...")

    for g_id in gesture_folders:
        folder_path = os.path.join(GESTURES_FOLDER, g_id)
        print(f"\nFlipping images for gesture {g_id}...")

        for i in range(1, TOTAL_IMAGES + 1):
            original_path = os.path.join(folder_path, f"{i}.jpg")
            flipped_path = os.path.join(folder_path, f"{i + TOTAL_IMAGES}.jpg")

            if not os.path.exists(original_path):
                print(f"Skipping missing: {original_path}")
                continue

            img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading: {original_path}")
                continue

            flipped_img = cv2.flip(img, 1)
            cv2.imwrite(flipped_path, flipped_img)

        print(f"Completed flipping for gesture {g_id}.")

    print("\nAll gestures processed successfully.")


# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    flip_images()
