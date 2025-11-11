"""
Hand Histogram Generator for Sign Language Recognition
------------------------------------------------------
Captures a clean hand color histogram using an external camera.
Use this histogram for accurate hand segmentation in gesture datasets.

Controls:
  c - capture histogram from sampling grid
  h - capture histogram from full ROI
  s - save and exit
"""

import cv2
import numpy as np
import pickle
import os


def get_external_camera():
    """Prefer external camera (index 1), fallback to 0 or 2."""
    print("Attempting to open external camera...")
    for idx in [1, 0, 2]:
        cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                print(f"Using camera index {idx}")
                return cam
        cam.release()
    print("No available camera found.")
    return None


def equalize_brightness(frame):
    """Apply CLAHE to reduce lighting variation."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def draw_sampling_grid(frame, start_x=360, start_y=120, box_w=25, box_h=25, gap=10):
    """Draw grid boxes and return stacked sampled pixels."""
    full_sample = None
    for i in range(6):
        row_sample = None
        for j in range(5):
            x, y = start_x + j * (box_w + gap), start_y + i * (box_h + gap)
            roi = frame[y:y + box_h, x:x + box_w]
            row_sample = roi if row_sample is None else np.hstack((row_sample, roi))
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)
        full_sample = row_sample if full_sample is None else np.vstack((full_sample, row_sample))
    return full_sample


def capture_hand_histogram():
    cam = get_external_camera()
    if cam is None:
        return

    print("Press 'c' (grid) | 'h' (ROI) | 's' (save & exit)")

    hist = None
    captured = False
    x, y, w, h = 280, 80, 340, 360

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        hand_sample = draw_sampling_grid(frame)

        if key == ord('c') and hand_sample is not None:
            hsv_sample = cv2.cvtColor(hand_sample, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_sample], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            captured = True
            print("Histogram captured from grid.")

        if key == ord('h'):
            roi = frame[y:y + h, x:x + w]
            hsv_sample = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_sample], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            captured = True
            print("Histogram captured from full ROI.")

        if captured and hist is not None:
            back_proj = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            lower_skin = np.array([0, 25, 30], dtype=np.uint8)
            upper_skin = np.array([179, 170, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            back_proj = cv2.bitwise_and(back_proj, back_proj, mask=skin_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            back_proj = cv2.filter2D(back_proj, -1, kernel)
            back_proj = cv2.GaussianBlur(back_proj, (7, 7), 0)
            back_proj = cv2.medianBlur(back_proj, 7)
            _, mask = cv2.threshold(back_proj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            clean_mask = np.zeros_like(mask)
            for cnt in contours:
                if cv2.contourArea(cnt) > 3000:
                    cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
            mask = clean_mask

            mask_display = cv2.merge((mask, mask, mask))
            cv2.imshow("Thresh", mask_display)

        cv2.imshow("Set Hand Histogram", frame)

        if key == ord('s'):
            print("Saving histogram and exiting...")
            break

    cam.release()
    cv2.destroyAllWindows()

    if hist is not None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hist")
        with open(save_path, "wb") as f:
            pickle.dump(hist, f)
        print(f"Histogram saved to: {save_path}")
    else:
        print("No histogram captured.")


if __name__ == "__main__":
    capture_hand_histogram()
