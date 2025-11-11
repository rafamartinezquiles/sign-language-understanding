"""
Hand Color Histogram Generator for Sign Language Recognition
-------------------------------------------------------------
This script helps you capture the color distribution (histogram)
of your hand in HSV space, which is later used for hand segmentation.

Steps:
1. Run this file.
2. Place your hand inside the green boxes on screen.
3. Press 'c' to capture the histogram of your hand skin tone.
4. Press 's' to save and exit.

Dependencies: OpenCV, NumPy, pickle
"""

import cv2
import numpy as np
import pickle


def draw_sampling_grid(frame):
    """
    Draws multiple small green boxes on the screen
    to sample pixels from different parts of the hand.
    
    Args:
        frame (numpy.ndarray): The current video frame.
    Returns:
        numpy.ndarray: The stacked cropped regions containing color samples.
    """
    # Starting coordinates (top-left corner)
    start_x, start_y = 420, 140
    box_w, box_h = 10, 10
    gap = 10

    # Initialize placeholders
    row_samples = None
    full_sample = None

    for row in range(10):
        for col in range(5):
            x1 = start_x + col * (box_w + gap)
            y1 = start_y + row * (box_h + gap)
            region = frame[y1:y1 + box_h, x1:x1 + box_w]

            # Stack boxes horizontally (row-wise)
            if row_samples is None:
                row_samples = region
            else:
                row_samples = np.hstack((row_samples, region))

            # Draw the rectangle grid on the frame
            cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 1)

        # Stack all rows vertically
        if full_sample is None:
            full_sample = row_samples
        else:
            full_sample = np.vstack((full_sample, row_samples))
        # Reset for next row
        row_samples = None  

    return full_sample


def capture_hand_histogram():
    """
    Captures and stores the HSV histogram of a user's hand skin tone
    using a webcam feed. The histogram can later be used for
    hand segmentation in gesture recognition models.
    """
    # Try opening the default camera (prefers external if available)
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    # Flags to manage capture flow
    histogram_captured = False
    histogram_saved = False
    hand_sample = None
    hand_histogram = None

    print("Press 'c' to capture hand histogram | 's' to save and exit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not access camera.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        key = cv2.waitKey(1) & 0xFF

        # Capture histogram when 'c' is pressed
        if key == ord('c') and hand_sample is not None:
            hsv_sample = cv2.cvtColor(hand_sample, cv2.COLOR_BGR2HSV)
            hand_histogram = cv2.calcHist([hsv_sample], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hand_histogram, hand_histogram, 0, 255, cv2.NORM_MINMAX)
            histogram_captured = True
            print("Histogram captured successfully.")

        # Exit when 's' is pressed
        elif key == ord('s'):
            histogram_saved = True
            print("Histogram saved. Exiting...")
            break

        # Once histogram captured, visualize backprojection
        if histogram_captured:
            back_proj = cv2.calcBackProject([hsv_frame], [0, 1], hand_histogram, [0, 180, 0, 256], 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(back_proj, -1, kernel, back_proj)
            blur = cv2.GaussianBlur(back_proj, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask_3ch = cv2.merge((mask, mask, mask))
            cv2.imshow("Hand Mask Preview", mask_3ch)

        # Draw sampling grid if histogram not yet saved
        if not histogram_saved:
            hand_sample = draw_sampling_grid(frame)

        cv2.imshow("Hand Histogram Setup", frame)

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

    # Save histogram
    if hand_histogram is not None:
        with open("hand_histogram.pkl", "wb") as f:
            pickle.dump(hand_histogram, f)
        print("Histogram file saved as 'hand_histogram.pkl'")


if __name__ == "__main__":
    capture_hand_histogram()
