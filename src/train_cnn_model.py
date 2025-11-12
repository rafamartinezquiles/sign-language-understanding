"""
Gesture CNN Trainer
-------------------
This script builds, trains, and evaluates a Convolutional Neural Network (CNN) for recognizing hand gestures captured in grayscale images.

Workflow:
1. Ensure your gesture dataset is prepared using the dataset preparer script:
   train_images, train_labels, val_images, val_labels
2. Ensure the "gestures" directory exists with subfolders named by gesture IDs:
   gestures/1/, gestures/2/, gestures/3/, etc.
3. The script:
   - Determines input image dimensions dynamically.
   - Builds a CNN with 3 convolutional layers.
   - Trains on the prepared dataset.
   - Automatically saves the best model to cnn_model_keras2.h5.

Outputs:
- Saved Keras model file: cnn_model_keras2.h5
- Console summary of training accuracy and final validation error.

Dependencies: TensorFlow / Keras, NumPy, OpenCV, Pickle, Glob, OS
"""

import os
import cv2
import numpy as np
import pickle
from glob import glob

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Silence TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ===============================================================
# Utility Functions
# ===============================================================

def get_image_size():
    """Detect image size dynamically from the first available gesture image."""
    gesture_dir = "gestures"
    if not os.path.exists(gesture_dir):
        raise FileNotFoundError(f"'{gesture_dir}' directory not found.")

    gesture_folders = sorted(
        [g for g in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, g))],
        key=lambda x: int(x) if x.isdigit() else x
    )
    for folder in gesture_folders:
        folder_path = os.path.join(gesture_dir, folder)
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(".jpg"):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img.shape
    raise FileNotFoundError("No valid gesture images found to determine input size.")


def get_num_of_classes():
    """Return the total number of gesture classes (subfolders in 'gestures/')."""
    gesture_dir = "gestures"
    if not os.path.exists(gesture_dir):
        raise FileNotFoundError(f"'{gesture_dir}' directory not found.")
    class_folders = [f for f in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, f))]
    return len(class_folders)


# ===============================================================
# CNN Model Definition
# ===============================================================

def build_cnn_model(image_x, image_y, num_classes):
    """Build and compile a CNN for gesture classification."""
    model = Sequential()

    # Convolutional layers with increasing filters and ReLU activation
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))

    # Fully connected classifier layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer and compilation
    optimizer = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Model checkpoint to save the best-performing model
    checkpoint_path = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    return model, [checkpoint]


# ===============================================================
# Training Routine
# ===============================================================

def train_cnn_model():
    """Train the CNN model using pre-saved gesture datasets."""
    required_files = ["train_images", "train_labels", "val_images", "val_labels"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing dataset file: '{file}'. Please run the dataset preparer script first.")

    # Load pickled data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # Get dynamic parameters
    image_x, image_y = get_image_size()
    num_classes = get_num_of_classes()

    # Reshape and normalize image data
    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

    train_images = train_images.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0

    # Adjust labels to start at 0 for Keras
    train_labels = train_labels - np.min(train_labels)
    val_labels = val_labels - np.min(val_labels)

    # One-hot encode labels using TensorFlow's to_categorical
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)

    # Display summary of dataset
    print(f"\nTraining dataset: {train_images.shape[0]} samples")
    print(f"Validation dataset: {val_images.shape[0]} samples")
    print(f"Input image size: {image_x}x{image_y}")
    print(f"Number of gesture classes: {num_classes}\n")

    # Build and train the model
    model, callbacks_list = build_cnn_model(image_x, image_y, num_classes)
    model.summary()

    model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=500,
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate final model performance
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\nValidation Accuracy: {scores[1] * 100:.2f}%")
    print(f"Validation Error: {100 - scores[1] * 100:.2f}%")

    K.clear_session()


# ===============================================================
# Main Execution
# ===============================================================
if __name__ == "__main__":
    train_cnn_model()
