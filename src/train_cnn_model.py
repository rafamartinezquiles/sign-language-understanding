"""
Gesture CNN Trainer
-------------------
This script builds, trains, and evaluates a Convolutional Neural Network (CNN)
for recognizing hand gestures captured in grayscale images.

Workflow:
1. Ensure your gesture dataset is prepared using the dataset preparer script:
       train_images, train_labels, val_images, val_labels
2. Ensure the "gestures" directory exists with subfolders named by gesture IDs:
       gestures/1/, gestures/2/, gestures/3/, etc.
3. The script:
   - Determines input image dimensions dynamically.
   - Builds a CNN with 3 convolutional layers.
   - Trains on the prepared dataset.
   - Automatically saves the best model to `cnn_model_keras2.h5`.

Outputs:
    - Saved Keras model file: cnn_model_keras2.h5
    - Console summary of training accuracy and final validation error.

Dependencies:
    TensorFlow / Keras, NumPy, OpenCV, Pickle, Glob, OS
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

# NEW: extra callbacks for better training control
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger  # NEW
from tensorflow.keras import backend as K

# NEW: (optional) metrics at the end
try:
    from sklearn.metrics import classification_report, confusion_matrix  # NEW
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Silence TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NEW: Reproducibility
import random  # NEW
import tensorflow as tf  # NEW
SEED = int(os.environ.get("SEED", 42))  # NEW
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)  # NEW


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

    class_folders = [f for f in os.listdir(gesture_dir)
                     if os.path.isdir(os.path.join(gesture_dir, f))]
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

    # NEW: keep SGD but make it stronger (momentum, nesterov, grad clipping)
    optimizer = optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True, clipnorm=1.0)  # CHANGED

    # NEW: enable light label smoothing for better generalization
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)  # NEW

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Model checkpoint to save the best-performing model
    checkpoint_path = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
    )

    # NEW: Add EarlyStopping and ReduceLROnPlateau + simple CSV log
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10,
                               restore_best_weights=True, verbose=1)  # NEW
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=4,
                                   factor=0.5, min_lr=1e-6, verbose=1)  # NEW
    csv_log = CSVLogger('training_log.csv', append=False)  # NEW

    # return all callbacks together
    return model, [checkpoint, early_stop, lr_plateau, csv_log]  # CHANGED


# ===============================================================
# Training Routine
# ===============================================================

def _compute_class_weights(y_int, num_classes):  # NEW
    """Compute simple inverse-frequency class weights."""
    counts = np.bincount(y_int, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts.astype(np.float32))
    return {i: float(weights[i]) for i in range(num_classes)}

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

    # NEW: optional dataset standardization (helps with lighting changes)
    mean = np.mean(train_images, axis=(0,1,2), keepdims=True)  # NEW
    std = np.std(train_images, axis=(0,1,2), keepdims=True) + 1e-7  # NEW
    train_images = (train_images - mean) / std  # NEW
    val_images = (val_images - mean) / std  # NEW

    # CHANGED: Use one global offset so train/val stay aligned
    global_min = min(int(train_labels.min()), int(val_labels.min()))  # NEW
    train_labels = train_labels - global_min  # CHANGED
    val_labels = val_labels - global_min  # CHANGED

    # NEW: Sanity-check label range vs num_classes
    if train_labels.max() >= num_classes or val_labels.max() >= num_classes:
        raise ValueError(
            f"Label index out of range. Max train={train_labels.max()}, "
            f"max val={val_labels.max()}, num_classes={num_classes}."
        )  # NEW

    # One-hot encode labels
    train_labels_1h = to_categorical(train_labels, num_classes)
    val_labels_1h = to_categorical(val_labels, num_classes)

    # Display summary of dataset
    print(f"\nTraining dataset: {train_images.shape[0]} samples")
    print(f"Validation dataset: {val_images.shape[0]} samples")
    print(f"Input image size: {image_x}x{image_y}")
    print(f"Number of gesture classes: {num_classes}\n")

    # Build and train the model
    model, callbacks_list = build_cnn_model(image_x, image_y, num_classes)
    model.summary()

    # NEW: handle class imbalance automatically (can be disabled by env var)
    use_class_weights = os.environ.get("USE_CLASS_WEIGHTS", "1") == "1"  # NEW
    class_weight = None  # NEW
    if use_class_weights:
        class_weight = _compute_class_weights(train_labels, num_classes)  # NEW
        print("Class weights:", class_weight)  # NEW

    # CHANGED: smaller, safer default batch size (still configurable)
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))  # CHANGED
    epochs = int(os.environ.get("EPOCHS", "100"))  # keep your 100 default

    history = model.fit(
        train_images, train_labels_1h,
        validation_data=(val_images, val_labels_1h),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True,                # NEW (explicit)
        class_weight=class_weight    # NEW (optional)
    )

    # Evaluate final model performance
    scores = model.evaluate(val_images, val_labels_1h, verbose=0)
    print(f"\nValidation Accuracy: {scores[1] * 100:.2f}%")
    print(f"Validation Error: {100 - scores[1] * 100:.2f}%")

    # NEW: optional per-class report & confusion matrix
    if _HAS_SKLEARN:
        preds = model.predict(val_images, verbose=0)
        y_true = val_labels
        y_pred = np.argmax(preds, axis=1)
        print("\nPer-class metrics:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))

    K.clear_session()


# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    train_cnn_model()
