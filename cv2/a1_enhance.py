
import cv2
import numpy as np


def enhance_images(normalized):
    """
    function called enhance_images() that takes an image path as input and returns an enhanced version of the image
    """

    enhanced=[]
    for result in normalized:
        result = result.astype(np.uint8)
        img = cv2.equalizeHist(result)
        enhanced.append(img)
    return enhanced
