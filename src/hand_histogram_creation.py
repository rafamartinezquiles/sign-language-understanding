"""
Advanced Hand Color Histogram Generator (Aligned Grid)
-------------------------------------------------------
Captures a clean HSV color histogram of your hand for gesture/sign recognition.

Key Improvements:
- Sampling grid fits perfectly inside the ROI box.
- Brightness equalization (CLAHE) for consistent color sampling.
- Combined HSV threshold + histogram segmentation.
- Morphological cleanup and filled contours for solid white hand masks.

Controls:
  C - Capture histogram from grid samples
  S - Save histogram and exit

Output:
  Saves 'hand_histogram.pkl' in this script’s folder.
"""

import cv2
import numpy as np
import pickle
import os


# ===============================================================
# Utility Functions
# ===============================================================

def equalize_brightness(img):
    """Normalize brightness using CLAHE in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def build_sampling_grid(frame, roi_x, roi_y, roi_w, roi_h, rows=10, cols=5):
    """
    Draws a grid of small green squares *inside* the ROI box.

    Returns:
        full_sample: Combined pixel sample from all boxes.
    """
    # Calculate margins and spacing
    gap_x = roi_w // (cols + 1)
    gap_y = roi_h // (rows + 1)
    box_w = int(gap_x * 0.6)
    box_h = int(gap_y * 0.6)

    full_sample = None

    for row in range(rows):
        row_sample = None
        for col in range(cols):
            # Center squares inside ROI
            x = roi_x + (col + 1) * gap_x - box_w // 2
            y = roi_y + (row + 1) * gap_y - box_h // 2

            roi_patch = frame[y:y + box_h, x:x + box_w]
            row_sample = roi_patch if row_sample is None else np.hstack((row_sample, roi_patch))

            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)

        full_sample = row_sample if full_sample is None else np.vstack((full_sample, row_sample))

    return full_sample


# ===============================================================
# Main Function
# ===============================================================

def get_hand_histogram():
    """Main loop for capturing and saving the hand color histogram."""
    # Try to open external camera first, fallback to internal
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: No camera detected.")
        return

    print("Press 'C' to capture histogram | 'S' to save and exit")

    hist = None
    histogram_captured = False

    # Define the Region Of Interest (ROI)
    roi_x, roi_y, roi_w, roi_h = 360, 120, 180, 240

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame = equalize_brightness(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        key = cv2.waitKey(1) & 0xFF

        # Draw ROI box
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

        # Build aligned sampling grid inside ROI
        sample_patch = build_sampling_grid(frame, roi_x, roi_y, roi_w, roi_h)

        # -------------------------------------------------------
        # 'C' pressed → capture histogram from grid
        # -------------------------------------------------------
        if key == ord('c') and sample_patch is not None:
            hsv_sample = cv2.cvtColor(sample_patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_sample], [0, 1], None,
                                [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            histogram_captured = True
            print("Histogram captured successfully.")

        # -------------------------------------------------------
        # Live segmentation (ROI only)
        # -------------------------------------------------------
        if histogram_captured and hist is not None:
            hsv_roi = hsv[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Backprojection with the histogram
            back_proj = cv2.calcBackProject([hsv_roi], [0, 1], hist,
                                            [0, 180, 0, 256], 1)

            # Restrict to typical skin HSV range
            lower_skin = np.array([0, 40, 70], dtype=np.uint8)
            upper_skin = np.array([20, 160, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

            # Combine histogram + HSV mask
            combined = cv2.bitwise_and(back_proj, back_proj, mask=skin_mask)

            # --- improved smoothing and threshold ---
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined = cv2.filter2D(combined, -1, kernel)
            combined = cv2.GaussianBlur(combined, (7, 7), 0)
            combined = cv2.medianBlur(combined, 5)

            _, mask = cv2.threshold(combined, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # --- fill small holes for solid hand ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled_mask = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > 2000:  # keep large contour (the hand)
                    cv2.drawContours(filled_mask, [cnt], -1, 255, cv2.FILLED)
            mask = cv2.dilate(filled_mask, kernel, iterations=2)

            # Display results
            cv2.imshow("Backprojection (raw)", back_proj)
            cv2.imshow("Segmentation Preview", mask)

        # -------------------------------------------------------
        # 'S' pressed → save and exit
        # -------------------------------------------------------
        if key == ord('s'):
            print("Saving histogram and exiting...")
            break

        cv2.imshow("Set Hand Histogram", frame)

    # -------------------------------------------------------
    # Cleanup and save histogram
    # -------------------------------------------------------
    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_histogram.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(hist, f)
        print(f"Histogram saved successfully at: {save_path}")
    else:
        print("No histogram captured; nothing saved.")


# ===============================================================
# Run Entry Point
# ===============================================================

if __name__ == "__main__":
    get_hand_histogram()
