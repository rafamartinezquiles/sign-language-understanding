"""
Gesture Data Collector for Sign Language Recognition
-----------------------------------------------------
This script captures hand gesture images using the webcam,
guided by a previously saved hand color histogram (for segmentation).

Features:
- Automatically creates folders and SQLite database in the same directory as this script.
- Stores gesture images under '<script_dir>/gestures/<gesture_id>/'.
- Each gesture gets an entry in '<script_dir>/gesture_db.db'.
- Uses hand color histogram for robust background separation.

Steps:
1. Make sure 'hand_histogram.pkl' exists in the same directory as this script.
2. Run this file.
3. Enter a unique gesture ID and name.
4. Place your hand in the green box and press 'c' to start capturing.
5. Press 'c' again to pause capturing, and it stops automatically after 1200 images.

Dependencies: OpenCV, NumPy, SQLite3, Pickle, OS, Random
"""

import cv2
import numpy as np
import os
import pickle
import random
import sqlite3

# Global constants for image dimensions
IMG_WIDTH, IMG_HEIGHT = 50, 50


def get_script_dir():
    """Return the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def load_hand_histogram():
    """Load the previously saved hand color histogram from this script's directory."""
    script_dir = get_script_dir()
    hist_path = os.path.join(script_dir, "hand_histogram.pkl")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(
            "Histogram file not found! Please run 'hand_histogram_generator.py' first."
        )

    with open(hist_path, "rb") as f:
        hand_hist = pickle.load(f)
    print("Hand histogram loaded successfully.")
    return hand_hist


def setup_environment():
    """Create gesture folder and database inside the script directory if not already present."""
    script_dir = get_script_dir()
    gestures_dir = os.path.join(script_dir, "gestures")
    db_path = os.path.join(script_dir, "gesture_db.db")

    if not os.path.exists(gestures_dir):
        os.mkdir(gestures_dir)
        print(f"Created '{gestures_dir}' directory.")

    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        create_table_query = """
        CREATE TABLE gesture (
            g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            g_name TEXT NOT NULL
        )
        """
        conn.execute(create_table_query)
        conn.commit()
        conn.close()
        print(f"Created SQLite database at '{db_path}'.")


def create_gesture_folder(gesture_id):
    """Create a new folder for storing gesture images inside the script directory."""
    script_dir = get_script_dir()
    folder_path = os.path.join(script_dir, "gestures", str(gesture_id))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Created folder for gesture ID {gesture_id}: {folder_path}")
    return folder_path


def save_gesture_to_db(gesture_id, gesture_name):
    """Insert or update gesture information in the database located in the script directory."""
    script_dir = get_script_dir()
    db_path = os.path.join(script_dir, "gesture_db.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (gesture_id, gesture_name)
        )
    except sqlite3.IntegrityError:
        choice = input(f"Gesture ID {gesture_id} already exists. Overwrite? (y/n): ").lower()
        if choice == "y":
            cursor.execute(
                "UPDATE gesture SET g_name = ? WHERE g_id = ?", (gesture_name, gesture_id)
            )
            print("Gesture record updated.")
        else:
            print("Operation cancelled.")
            conn.close()
            return
    conn.commit()
    conn.close()


def capture_gesture_images(gesture_id, hand_hist):
    """
    Capture images of a specific gesture using webcam and save them
    inside the script's folder structure.
    """
    total_images = 1200
    captured_count = 0
    capture_active = False
    frame_counter = 0

    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    # ROI (Region of Interest) rectangle
    roi_x, roi_y, roi_w, roi_h = 300, 100, 300, 300
    folder_path = create_gesture_folder(gesture_id)

    print("Press 'c' to start/pause capturing | Captures 1200 images automatically")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not read from camera.")
            break

        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply histogram backprojection for skin segmentation
        back_proj = cv2.calcBackProject([hsv_frame], [0, 1], hand_hist, [0, 180, 0, 256], 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(back_proj, -1, kernel, back_proj)
        blur = cv2.GaussianBlur(back_proj, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray_mask = cv2.merge((thresh, thresh, thresh))
        gray_mask = cv2.cvtColor(gray_mask, cv2.COLOR_BGR2GRAY)
        gray_mask = gray_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        contours, _ = cv2.findContours(gray_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 10000 and frame_counter > 50:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped = gray_mask[y:y + h, x:x + w]

                # Make the cropped image square
                if w > h:
                    pad = (w - h) // 2
                    cropped = cv2.copyMakeBorder(cropped, pad, pad, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h > w:
                    pad = (h - w) // 2
                    cropped = cv2.copyMakeBorder(cropped, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))

                # Resize to target dimensions
                final_img = cv2.resize(cropped, (IMG_WIDTH, IMG_HEIGHT))

                # Randomly augment (flip horizontally)
                if random.randint(0, 1) == 1:
                    final_img = cv2.flip(final_img, 1)

                # Save image
                captured_count += 1
                img_path = os.path.join(folder_path, f"{captured_count}.jpg")
                cv2.imwrite(img_path, final_img)
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.8, (127, 255, 255), 2)

        # Display ROI and progress
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {captured_count}/{total_images}", (30, 400),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 127, 255), 2)
        cv2.imshow("Gesture Capture", frame)
        cv2.imshow("Threshold", gray_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            capture_active = not capture_active
            frame_counter = 0
        if capture_active:
            frame_counter += 1
        if captured_count >= total_images:
            print(f"Successfully captured {total_images} images for gesture {gesture_id}.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_environment()

    gesture_id = input("Enter gesture ID (number): ")
    gesture_name = input("Enter gesture name/label: ")

    save_gesture_to_db(gesture_id, gesture_name)
    histogram = load_hand_histogram()
    capture_gesture_images(gesture_id, histogram)
