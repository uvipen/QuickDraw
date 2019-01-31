import cv2
import numpy as np

drawing = False  # true if mouse is pressed


# mouse callback function
def paint_draw(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(image, (ix, iy), (x, y), (255, 255, 255), 5)
            ix = x
            iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(image, (ix, iy), (x, y), (255, 255, 255), 5)
        ix = x
        iy = y
    return x, y


image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Canvas")
cv2.setMouseCallback('Canvas', paint_draw)
while (1):
    cv2.imshow('Canvas', 255-image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Escape KEY
        cv2.imwrite("painted_image.jpg", image)
        break
cv2.destroyAllWindows()