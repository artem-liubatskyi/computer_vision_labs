# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# read images
bright = cv2.imread("./assets/bright.jpg")
dark = cv2.imread("./assets/dark.jpg")


def apply_filter(image, filter: int):
    bgr = [40, 158, 16]
    thresh = 40

    filteredImage = image
    filteredBgr = bgr

    if filter is not None:
        filteredImage = cv2.cvtColor(image, filter)
        filteredBgr = cv2.cvtColor(
            np.uint8([[bgr]]), filter)[0][0]

    min = np.array([filteredBgr[0] - thresh,
                   filteredBgr[1] - thresh, filteredBgr[2] - thresh])
    max = np.array([filteredBgr[0] + thresh,
                   filteredBgr[1] + thresh, filteredBgr[2] + thresh])

    mask = cv2.inRange(filteredImage, min, max)
    return cv2.bitwise_and(filteredImage, filteredImage, mask=mask)


labels = ["Result BGR", "Result HSV", "Result YCB", "Result LAB"]
filters = [None, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2YCrCb, cv2.COLOR_BGR2LAB]

figs, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, filter in enumerate(filters):
    axes[0][i].imshow(apply_filter(bright, filter))
    axes[0][i].set_title("Bright - " + labels[i])

    axes[1][i].imshow(apply_filter(dark, filter))
    axes[1][i].set_title("Dark - " + labels[i])

plt.show()
