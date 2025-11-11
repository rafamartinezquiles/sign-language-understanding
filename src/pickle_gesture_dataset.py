"""
Gesture Dataset Preparer
------------------------
This script loads all gesture images, extracts their labels,
and splits them into training, testing, and validation sets.

Workflow:
1. The "gestures" directory should contain subfolders named by gesture IDs.
   Example: gestures/1/, gestures/2/, gestures/3/, etc.
2. Each subfolder should contain grayscale .jpg gesture images.
3. The script:
   - Reads all images and their labels.
   - Shuffles the dataset thoroughly.
   - Splits into 5/6 training, 1/12 testing, 1/12 validation.
   - Saves the splits as pickle files for later model training.

Outputs:
    train_images, train_labels
    test_images, test_labels
    val_images, val_labels

Dependencies:
    OpenCV, NumPy, scikit-learn, Pickle, OS, Glob
"""

import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os


# ===============================================================
# Core Functionality
# ===============================================================

def pickle_images_labels():
    """Load gesture images, assign numeric labels, and return as a list of tuples."""
    images_labels = []
    images = glob(os.path.join("gestures", "*", "*.jpg"))
    images.sort()

    if not images:
        print("No gesture images found. Please ensure the 'gestures/' directory contains labeled subfolders.")
        return []

    print(f"Found {len(images)} gesture images. Loading and labeling...")

    for image in images:
        label = image[image.find(os.sep) + 1: image.rfind(os.sep)]
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: could not read {image}. Skipping.")
            continue

        images_labels.append((np.array(img, dtype=np.uint8), int(label)))

    print(f"Loaded {len(images_labels)} valid images.")
    return images_labels


# ===============================================================
# Dataset Preparation
# ===============================================================

def create_dataset_splits():
    """Shuffle, split, and save gesture images and labels into pickle files."""
    images_labels = pickle_images_labels()
    if not images_labels:
        print("Dataset preparation aborted: no images found.")
        return

    # Shuffle multiple times for randomness
    for _ in range(4):
        images_labels = shuffle(images_labels)

    images, labels = zip(*images_labels)
    total = len(images_labels)
    print(f"Total images: {total}")

    # Compute split indices
    train_end = int(5/6 * total)
    test_end = int(11/12 * total)

    # Split dataset
    train_images, train_labels = images[:train_end], labels[:train_end]
    test_images, test_labels = images[train_end:test_end], labels[train_end:test_end]
    val_images, val_labels = images[test_end:], labels[test_end:]

    # Save all splits
    save_pickle("train_images", train_images)
    save_pickle("train_labels", train_labels)
    save_pickle("test_images", test_images)
    save_pickle("test_labels", test_labels)
    save_pickle("val_images", val_images)
    save_pickle("val_labels", val_labels)

    print("\nDataset successfully split and saved:")
    print(f"  Train set: {len(train_images)} images")
    print(f"  Test set:  {len(test_images)} images")
    print(f"  Val set:   {len(val_images)} images")


# ===============================================================
# Helper Function
# ===============================================================

def save_pickle(filename, data):
    """Save data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved: {filename} ({len(data)} items)")


# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    create_dataset_splits()
