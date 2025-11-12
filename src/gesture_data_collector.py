"""
Advanced Gesture Data Collector
--------------------------------
Captures segmented hand images for sign/gesture recognition training.

This script uses the previously saved 'hand_histogram.pkl' to segment
your hand and automatically captures gesture images into folders.

Workflow:
1. Ensure 'hand_histogram.pkl' exists (created by the histogram generator).
2. Run this script.
3. Enter gesture ID and name.
4. Place your hand inside the ROI (green box).
5. Press 'C' to start capturing, 'C' again to pause.
6. Automatically saves 1000 processed gesture images to:
       gestures/<gesture_id>/

Features:
- CLAHE lighting normalization
- Robust histogram-based skin segmentation
- Smooth, filled, solid white-hand masks
- SQLite database integration for gesture metadata

Dependencies:
    OpenCV, NumPy, Pickle, SQLite3
"""

import cv2
import numpy as np
import os
import pickle
import random
import sqlite3


# ===============================================================
# Global Parameters
# ===============================================================

IMG_WIDTH, IMG_HEIGHT = 50, 50
ROI_X, ROI_Y, ROI_W, ROI_H = 360, 120, 180, 240
TOTAL_IMAGES = 1000


# ===============================================================
# Utility Functions
# ===============================================================

def equalize_brightness(frame):
    """Apply CLAHE to reduce lighting variations."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_hand_histogram():
    """Load the previously saved hand histogram from the same folder as this script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hist_path = os.path.join(current_dir, "hand_histogram.pkl")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            f"Histogram file not found in: {hist_path}\n"
            f"Please run 'hand_histogram_creation.py' first to generate it."
        )

    with open(hist_path, "rb") as f:
        return pickle.load(f)


def setup_environment():
    """Create the gestures folder and SQLite database if missing."""
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
        print("Created 'gestures' directory.")

    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        conn.execute("""
        CREATE TABLE gesture (
            g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            g_name TEXT NOT NULL
        )
        """)
        conn.commit()
        conn.close()
        print("Created SQLite database 'gesture_db.db'.")


def create_gesture_folder(gesture_id):
    """Create a dedicated folder for this gesture's images."""
    folder = os.path.join("gestures", str(gesture_id))
    if not os.path.exists(folder):
        os.mkdir(folder)
        print(f"Created folder for gesture {gesture_id}.")
    return folder


def save_gesture_to_db(gesture_id, gesture_name):
    """Insert or update gesture info in SQLite DB."""
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (gesture_id, gesture_name))
    except sqlite3.IntegrityError:
        choice = input(f"Gesture ID {gesture_id} already exists. Overwrite? (y/n): ").lower()
        if choice == "y":
            cursor.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (gesture_name, gesture_id))
            print("Gesture record updated.")
        else:
            print("Operation cancelled.")
            conn.close()
            return
    conn.commit()
    conn.close()


# ===============================================================
# Core Functionality
# ===============================================================

def capture_gesture_images(gesture_id, hand_hist):
    """
    Capture gesture images using a webcam and hand segmentation.
    """
    folder_path = create_gesture_folder(gesture_id)
    total_captured = 0
    capturing = False
    frame_counter = 0

    # Initialize camera (prefer external)
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("Press 'C' to start/pause capturing | Press 'Q' to quit early")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame = equalize_brightness(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ROI region
        roi = hsv[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

        # Backprojection segmentation
        back_proj = cv2.calcBackProject([roi], [0, 1], hand_hist, [0, 180, 0, 256], 1)

        # HSV skin-range mask
        lower_skin = np.array([0, 40, 70], dtype=np.uint8)
        upper_skin = np.array([20, 160, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(roi, lower_skin, upper_skin)

        # Combine and clean
        combined = cv2.bitwise_and(back_proj, back_proj, mask=skin_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.filter2D(combined, -1, kernel)
        combined = cv2.GaussianBlur(combined, (7, 7), 0)
        combined = cv2.medianBlur(combined, 5)

        # Threshold + Morphological cleanup
        _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Fill hand shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:
                cv2.drawContours(filled_mask, [cnt], -1, 255, cv2.FILLED)
        mask = cv2.dilate(filled_mask, kernel, iterations=2)

        # Capture and save
        if capturing and contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frame_counter > 30:
                x, y, w, h = cv2.boundingRect(contour)
                crop = mask[y:y + h, x:x + w]

                # Make square
                if w > h:
                    pad = (w - h) // 2
                    crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                elif h > w:
                    pad = (h - w) // 2
                    crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

                # Resize and random flip
                final_img = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))
                if random.randint(0, 1):
                    final_img = cv2.flip(final_img, 1)

                total_captured += 1
                img_path = os.path.join(folder_path, f"{total_captured}.jpg")
                cv2.imwrite(img_path, final_img)
                cv2.putText(frame, "Capturing...", (30, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 255, 255), 2)

        # Draw ROI and show feedback
        cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {total_captured}/{TOTAL_IMAGES}",
                    (20, 440), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (127, 127, 255), 2)

        cv2.imshow("Gesture Capture", frame)
        cv2.imshow("Mask", mask)

        # Handle keypresses
        key = cv2.waitKey(10) & 0xFF

        if key in [ord('c'), ord('C')]:
            capturing = not capturing
            frame_counter = 0
            state = "STARTED" if capturing else "PAUSED"
            print(f"Capture {state} (press 'C' again to toggle)")

        elif key in [ord('q'), ord('Q')]:
            print(f"Exiting early after {total_captured} images.")
            break

        # Increment frame counter only while capturing
        if capturing:
            frame_counter += 1

        # Stop automatically once TOTAL_IMAGES are captured
        if total_captured >= TOTAL_IMAGES:
            print(f"Finished capturing {total_captured} images for gesture {gesture_id}.")
            break

    cam.release()
    cv2.destroyAllWindows()


# ===============================================================
# Main Execution
# ===============================================================

if __name__ == "__main__":
    setup_environment()

    gesture_id = input("Enter gesture ID (number): ")
    gesture_name = input("Enter gesture name/label: ")

    save_gesture_to_db(gesture_id, gesture_name)
    hand_hist = load_hand_histogram()
    capture_gesture_images(gesture_id, hand_hist)
