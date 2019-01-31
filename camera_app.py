"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

# from src.dataset import CLASSES
from src.config import *
from src.utils import get_images, get_overlay


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-c", "--color", type=str, choices=["green", "blue", "red"], default="green",
                        help="Color which could be captured by camera and seen as pointer")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-d", "--display", type=int, default=3, help="How long is prediction shown in second(s)")
    parser.add_argument("-s", "--canvas", type=bool, default=False, help="Display black & white canvas")
    args = parser.parse_args()
    return args


def main(opt):
    # Define color range
    if opt.color == "red":  # We shouldn't use red as color for pointer, since it
        # could be confused with our skin's color under some circumstances
        color_lower = np.array(RED_HSV_LOWER)
        color_upper = np.array(RED_HSV_UPPER)
        color_pointer = RED_RGB
    elif opt.color == "green":
        color_lower = np.array(GREEN_HSV_LOWER)
        color_upper = np.array(GREEN_HSV_UPPER)
        color_pointer = GREEN_RGB
    else:
        color_lower = np.array(BLUE_HSV_LOWER)
        color_upper = np.array(BLUE_HSV_UPPER)
        color_pointer = BLUE_RGB

    # Initialize deque for storing detected points and canvas for drawing
    points = deque(maxlen=512)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Load the video from camera (Here I use built-in webcam)
    camera = cv2.VideoCapture(0)
    is_drawing = False
    is_shown = False

    # Load images for classes:
    class_images = get_images("images", CLASSES)
    predicted_class = None

    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=512)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    # Check if the largest contour satisfy the condition of minimum area
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, None, :, :]
                        image = torch.from_numpy(image)
                        logits = model(image)
                        predicted_class = torch.argmax(logits[0])
                        # print (CLASSES[predicted_class])
                        is_shown = True
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=512)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read frame from camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        # Detect pixels fall within the pre-defined color range. Then, blur the image
        mask = cv2.inRange(hsv, color_lower, color_upper)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check to see if any contours are found
        if len(contours):
            # Take the biggest contour, since it is possible that there are other objects in front of camera
            # whose color falls within the range of our pre-defined color
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), YELLOW_RGB, 2)
            if is_drawing:
                M = cv2.moments(contour)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                points.appendleft(center)
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1], points[i], WHITE_RGB, 5)
                    cv2.line(frame, points[i - 1], points[i], color_pointer, 2)

        if is_shown:
            cv2.putText(frame, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_pointer, 5, cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60,60))


        cv2.imshow("Camera", frame)
        if opt.canvas:
            cv2.imshow("Canvas", 255-canvas)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
