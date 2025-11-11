"""
Gesture Dataset Preparation Script
----------------------------------
This script processes all gesture images, assigns class labels, 
and splits them into training, testing, and validation datasets.
Each dataset is serialized into pickle files for easy loading 
during CNN model training.

Features:
- Automatically finds all gesture images inside the 'gestures' folder.
- Extracts labels from folder names.
- Shuffles data multiple times for randomness.
- Splits data into 5/6 training, 1/12 testing, and 1/12 validation.
- Saves all pickle files in the same folder as this script.

Dependencies: OpenCV, NumPy, scikit-learn, pickle, glob, os
"""

import cv2
import numpy as np
import os
import pickle
from glob import glob
from sklearn.utils import shuffle


# ----------------------------------------
# Helper Functions
# ----------------------------------------
def load_all_images():
    """
    Loads grayscale gesture images and their numeric labels.

    Returns:
        list of (image_array, label_int) tuples.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gestures_dir = os.path.join(script_dir, "gestures")

    if not os.path.exists(gestures_dir):
        raise FileNotFoundError("'gestures' folder not found. Please collect gestures first.")

    image_label_pairs = []
    image_paths = sorted(glob(os.path.join(gestures_dir, "*", "*.jpg")))

    print(f"Found {len(image_paths)} images across gesture classes.")

    for img_path in image_paths:
        # Extract label (folder name after gestures/)
        label_name = os.path.basename(os.path.dirname(img_path))
        try:
            label = int(label_name)  # label folders are numeric
        except ValueError:
            print(f"Skipping non-numeric folder: {label_name}")
            continue

        # Read image in grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        image_label_pairs.append((np.array(img_gray, dtype=np.uint8), label))

    print(f"Loaded {len(image_label_pairs)} valid image-label pairs.")
    return image_label_pairs


def save_pickle(obj, filename):
    """Saves a Python object as a pickle file in the same folder as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {save_path} ({len(obj)} entries)")


# ----------------------------------------
# Main Dataset Preparation Logic
# ----------------------------------------
def prepare_datasets():
    """Loads gesture images, shuffles, splits, and pickles them into train/test/val sets."""
    all_data = load_all_images()

    # Shuffle multiple times for stronger randomization
    for _ in range(4):
        all_data = shuffle(all_data, random_state=None)

    # Unzip images and labels
    images, labels = zip(*all_data)
    total_count = len(images)
    print(f"\nTotal samples: {total_count}")

    # Split ratios
    train_end = int(5 / 6 * total_count)
    test_end = int(11 / 12 * total_count)

    # --- Training set ---
    train_images, train_labels = images[:train_end], labels[:train_end]
    save_pickle(train_images, "train_images.pkl")
    save_pickle(train_labels, "train_labels.pkl")

    # --- Testing set ---
    test_images, test_labels = images[train_end:test_end], labels[train_end:test_end]
    save_pickle(test_images, "test_images.pkl")
    save_pickle(test_labels, "test_labels.pkl")

    # --- Validation set ---
    val_images, val_labels = images[test_end:], labels[test_end:]
    save_pickle(val_images, "val_images.pkl")
    save_pickle(val_labels, "val_labels.pkl")

    print("\nDataset preparation complete.")
    print(f"  • Training samples: {len(train_images)}")
    print(f"  • Testing samples:  {len(test_images)}")
    print(f"  • Validation samples: {len(val_images)}")


# ----------------------------------------
# Script Entry Point
# ----------------------------------------
if __name__ == "__main__":
    prepare_datasets()
