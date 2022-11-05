import cv2
from matplotlib import pyplot as plt
from skimage.filters import gaussian, median
from skimage.morphology import disk
from skimage.transform import rescale
from skimage.util import random_noise

img1 = cv2.imread('./assets/histogram_1.jpg', cv2.IMREAD_GRAYSCALE)
ing1 = rescale(img1, 0.15)

img_ns = random_noise(img1, mode='speckle', mean=0.1)

img_m3 = median(img_ns, disk(3))
img_m9 = median(img_ns, disk(11))

plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original image')
plt.subplot(2, 2, 2)
plt.imshow(img_ns, cmap='gray')
plt.title('Noisy image')
plt.subplot(2, 2, 3)
plt.imshow(img_m3, cmap='gray')
plt.title("Median filter (radius=3) ")
plt.subplot(2, 2, 4)
plt.imshow(img_m9, cmap='gray')
plt.title('Median filter(radius-9)')
plt.show()

img_g1 = gaussian(img_ns, 1)
img_g3 = gaussian(img_ns, 3)

plt.subplot(2, 2, 1)
plt.show(img1, cmap='gray')
plt.title('Original image')
plt.subplot(2, 2, 2)
plt.imshow(img_ns, cmap='gray')
plt.title('Noisy Image')
plt.subplot(2, 2, 3)
plt.imshow(img_g1, cmap='gray')
plt.title("Gaussian filter (sigma-1)")
plt.subplot(2, 2, 4)
plt.imshow(img_g3, cmap='gray')
plt.title('Gaussian filter(sigma 3)')
plt.show()
