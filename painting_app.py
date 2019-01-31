import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch



def main():
    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
    model.eval()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.namedWindow("Canvas")
    global ix, iy, is_drawing
    is_drawing = False

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing == True:
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        elif event == cv2.EVENT_LBUTTONUP:
            is_drawing = False
            cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y

    cv2.setMouseCallback('Canvas', paint_draw)
    while (1):
        cv2.imshow('Canvas', 255 - image)
        key = cv2.waitKey(10)
        if key == ord(" "):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(image)
            min_y = np.min(ys)
            max_y = np.max(ys)
            min_x = np.min(xs)
            max_x = np.max(xs)
            image = image[min_y:max_y, min_x: max_x]

            image = cv2.resize(image, (28, 28))
            image = np.array(image, dtype=np.float32)[None, None, :, :]
            image = torch.from_numpy(image)
            logits = model(image)
            print(CLASSES[torch.argmax(logits[0])])
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            ix = -1
            iy = -1







if __name__ == '__main__':
    main()
