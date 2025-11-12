"""
Gesture Recognizer & Calculator
-------------------------------
Real-time gesture recognition with two modes:
  1) Text Mode       - speaks recognized words
  2) Calculator Mode - performs arithmetic via gestures

Hotkeys:
  'q' = quit current mode
  't' = switch to Text Mode
  'c' = switch to Calculator Mode
  'v' = toggle voice on/off

Requirements in the repository:
  - Trained model:        cnn_model_keras2.h5
  - Histogram file:       hand_histogram.pkl  (or 'hist', 'hand_histogram')
  - SQLite gestures DB:   gesture_db.db       (table 'gesture' with g_id, g_name)
  - gestures/<id>/ images (only used to infer image size)
"""

import os
import cv2
import pickle
import sqlite3
import pyttsx3
import numpy as np
from threading import Thread
from tensorflow.keras.models import load_model

# ===============================================================
# Configuration
# ===============================================================

MODEL_PATH = "cnn_model_keras2.h5"

# Use the SAME ROI as your histogram generator
ROI_X, ROI_Y, ROI_W, ROI_H = 360, 120, 180, 240

# ===============================================================
# Initialization
# ===============================================================

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Quieter TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load CNN
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

def load_hand_histogram():
    """Load the pre-saved hand color histogram (pkl or raw pickle)."""
    candidates = ["hand_histogram.pkl", "hand_histogram", "hist"]
    base = os.path.dirname(__file__)
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        "Histogram file not found. Expected one of: 'hand_histogram.pkl', 'hand_histogram', or 'hist'. "
        "Please run your histogram generator first."
    )

hist = load_hand_histogram()
is_voice_on = True

# ===============================================================
# Utilities shared with your generator
# ===============================================================

def equalize_brightness(bgr):
    """CLAHE on L channel in LAB to stabilize HSV under lighting changes."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def get_image_size():
    """Infer input image size from the first valid grayscale image in gestures/."""
    base_dir = "gestures"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"'{base_dir}' directory not found.")

    for folder in sorted(os.listdir(base_dir), key=lambda x: int(x) if x.isdigit() else x):
        fpath = os.path.join(base_dir, folder)
        if not os.path.isdir(fpath):
            continue
        for name in os.listdir(fpath):
            if name.lower().endswith(".jpg"):
                img = cv2.imread(os.path.join(fpath, name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img.shape
    raise FileNotFoundError("No sample gesture image found to determine input size.")

image_x, image_y = get_image_size()

def preprocess_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = img.astype(np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(img_gray):
    """Return (probability, class_id) for a grayscale image."""
    pred = model.predict(preprocess_image(img_gray), verbose=0)[0]
    cls = int(np.argmax(pred))
    return float(np.max(pred)), cls

def get_label_from_db(class_id):
    conn = sqlite3.connect("gesture_db.db")
    cur = conn.execute("SELECT g_name FROM gesture WHERE g_id = ?", (class_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "Unknown"

def say_text(text):
    """Non-blocking text-to-speech, guarded by is_voice_on flag."""
    global is_voice_on
    if not is_voice_on:
        return
    while getattr(engine, "_inLoop", False):
        pass
    engine.say(text)
    engine.runAndWait()

# ===============================================================
# Camera helpers (reduce blur)
# ===============================================================

def open_camera():
    """
    Prefer external cam index 1; fall back to 0.
    Request 640x480 and try to disable autofocus if supported
    (prevents focus hunting blur on some Windows drivers).
    """
    for idx in (1, 0):
        cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cam.isOpened():
            # Request resolution
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Try to disable autofocus and set a mid focus value if supported
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0=off, 1=on (if driver supports)
            cam.set(cv2.CAP_PROP_FOCUS, 0)      # some drivers accept 0..255 or 0..1

            # A tiny sharpen to counter mild defocus from driver scaling
            return cam
    return None

# ===============================================================
# Segmentation (aligned with your histogram generator)
# ===============================================================

def segment_hand_roi(frame_bgr):
    """
    Perform segmentation ONLY on the ROI using:
      - CLAHE brightness normalization
      - HSV backprojection with saved histogram
      - Tight skin-range HSV mask
      - Morphological cleanup + filled contours
    Returns: (display_frame, filled_mask_in_roi, contours)
    """
    # Flip like the generator
    frame_bgr = cv2.flip(frame_bgr, 1)

    # Resize once for consistency
    frame_bgr = cv2.resize(frame_bgr, (640, 480))

    # Brightness normalization (CLAHE)
    frame_eq = equalize_brightness(frame_bgr)

    # HSV
    hsv = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2HSV)

    # ROI slice
    roi = hsv[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Backprojection using histogram
    back_proj = cv2.calcBackProject([roi], [0, 1], hist, [0, 180, 0, 256], 1)

    # Tight skin HSV range (same as generator you pasted)
    lower_skin = np.array([0, 40, 70], dtype=np.uint8)
    upper_skin = np.array([20, 160, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(roi, lower_skin, upper_skin)

    # Combine and smooth
    combined = cv2.bitwise_and(back_proj, back_proj, mask=skin_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.filter2D(combined, -1, kernel)
    combined = cv2.GaussianBlur(combined, (7, 7), 0)
    combined = cv2.medianBlur(combined, 5)

    # Binary mask
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morph cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill big contours (hand)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)

    # Optional dilation for solidity
    filled = cv2.dilate(filled, kernel, iterations=2)

    # Draw ROI on the display frame
    disp = frame_bgr.copy()
    cv2.rectangle(disp, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
    return disp, filled, contours

def predict_from_contour(contour, roi_mask):
    """Crop from ROI mask by contour and classify."""
    x, y, w, h = cv2.boundingRect(contour)
    crop = roi_mask[y:y + h, x:x + w]

    # Make square
    if w > h:
        pad = (w - h) // 2
        crop = cv2.copyMakeBorder(crop, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h > w:
        pad = (h - w) // 2
        crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

    prob, cls = keras_predict(crop)
    if prob * 100 > 70:
        return get_label_from_db(cls)
    return ""

def get_operator(pred_text):
    """Map digit gestures to operators for calculator mode."""
    mapping = {0: "|", 1: "+", 2: "-", 3: "*", 4: "/", 5: "%", 6: "**", 7: ">>", 8: "<<", 9: "&"}
    try:
        return mapping.get(int(pred_text), "")
    except ValueError:
        return ""

# ===============================================================
# Modes
# ===============================================================

def calculator_mode(cam):
    global is_voice_on
    flags = {"first": False, "operator": False, "second": False, "clear": False}
    same_frames = 0
    first = operator = second = pred_text = calc_text = ""
    info = "Enter first number"
    Thread(target=say_text, args=(info,)).start()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        disp, roi_mask, contours = segment_hand_roi(frame)
        old = pred_text

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10000:
                pred_text = predict_from_contour(c, roi_mask)
                same_frames = same_frames + 1 if pred_text == old else 0

                # Clear
                if pred_text == "C" and same_frames > 5:
                    first = operator = second = pred_text = calc_text = ""
                    for k in flags: flags[k] = False
                    info = "Enter first number"
                    Thread(target=say_text, args=(info,)).start()
                    same_frames = 0

                # Execute "Best of Luck "
                elif pred_text == "Best of Luck " and same_frames > 15:
                    same_frames = 0
                    if flags["clear"]:
                        first = operator = second = pred_text = calc_text = ""
                        for k in flags: flags[k] = False
                        info = "Enter first number"
                        Thread(target=say_text, args=(info,)).start()
                    elif second:
                        flags["second"] = True
                        flags["clear"] = True
                        try:
                            calc_text += " = " + str(eval(calc_text))
                        except Exception:
                            calc_text = "Invalid operation"
                        Thread(target=say_text, args=(calc_text,)).start()
                    elif first:
                        flags["first"] = True
                        info = "Enter operator"
                        Thread(target=say_text, args=(info,)).start()

                # Digits
                elif pred_text.isnumeric():
                    if not flags["first"] and same_frames > 15:
                        first += pred_text
                        calc_text += pred_text
                        Thread(target=say_text, args=(pred_text,)).start()
                        same_frames = 0
                    elif not flags["operator"]:
                        op = get_operator(pred_text)
                        if op and same_frames > 15:
                            calc_text += op
                            flags["operator"] = True
                            info = "Enter second number"
                            Thread(target=say_text, args=(info,)).start()
                            same_frames = 0
                    elif not flags["second"] and same_frames > 15:
                        second += pred_text
                        calc_text += pred_text
                        Thread(target=say_text, args=(pred_text,)).start()
                        same_frames = 0

        # Draw UI
        board = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(board, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0))
        cv2.putText(board, f"Predicted: {pred_text}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0))
        cv2.putText(board, calc_text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255))
        cv2.putText(board, info, (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))
        out = np.hstack((disp, board))
        cv2.imshow("Recognizing gesture", out)
        cv2.imshow("Threshold", roi_mask)

        key = cv2.waitKey(1)
        if key in (ord("q"), ord("t")):
            return 1 if key == ord("t") else 0
        if key == ord("v"):
            is_voice_on = not is_voice_on

def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    same = 0

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        disp, roi_mask, contours = segment_hand_roi(frame)
        old = text

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10000:
                text = predict_from_contour(c, roi_mask)
                same = same + 1 if old == text else 0

                if same > 20 and text:
                    Thread(target=say_text, args=(text,)).start()
                    word += text
                    same = 0
            elif cv2.contourArea(c) < 1000 and word:
                Thread(target=say_text, args=(word,)).start()
                word = ""
        else:
            if word:
                Thread(target=say_text, args=(word,)).start()
                word = ""

        board = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(board, "Text Mode", (200, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0))
        cv2.putText(board, f"Predicted: {text}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0))
        cv2.putText(board, word, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255))
        out = np.hstack((disp, board))
        cv2.imshow("Recognizing gesture", out)
        cv2.imshow("Threshold", roi_mask)

        key = cv2.waitKey(1)
        if key in (ord("q"), ord("c")):
            return 2 if key == ord("c") else 0
        if key == ord("v"):
            is_voice_on = not is_voice_on

# ===============================================================
# Main
# ===============================================================

def recognize():
    cam = open_camera()
    if cam is None or not cam.isOpened():
        # final fallback
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise RuntimeError("No camera available.")

    # warm-up predict once
    _ = model.predict(np.zeros((1, image_x, image_y, 1), dtype=np.float32), verbose=0)

    keypress = 1
    while True:
        if keypress == 1:
            keypress = text_mode(cam)
        elif keypress == 2:
            keypress = calculator_mode(cam)
        else:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
