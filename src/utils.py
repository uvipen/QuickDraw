"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import cv2
import numpy as np
from sklearn import metrics


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def get_images(path, classes):
    images = [cv2.imread("{}/{}.png".format(path, item), cv2.IMREAD_UNCHANGED) for item in classes]
    return images


def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    fg_image = cv2.resize(fg_image, sizes)
    fg_mask = fg_image[:, :, 3:]
    fg_image = fg_image[:, :, :3]
    bg_mask = 255 - fg_mask
    bg_image = bg_image/255
    fg_image = fg_image/255
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)/255
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)/255
    image = cv2.addWeighted(bg_image*bg_mask, 255, fg_image*fg_mask, 255, 0.).astype(np.uint8)
    return image





# if __name__ == '__main__':
#     images = get_images("../images", ["apple", "star"])
#     print(images[0].shape)
#     print(np.max(images[0]))
