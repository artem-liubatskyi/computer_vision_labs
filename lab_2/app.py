import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops

PATCH_SIZE = 34


def createPatches(locations, patchSize=PATCH_SIZE):
    patches = []
    for loc in locations:
        patches.append(image[loc[0]:loc[0] + patchSize,
                             loc[1]:loc[1] + patchSize])
    return patches


def renderPatches(patches, label_prefix, offset):
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(3, len(patches), len(patches)*offset + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
        ax.set_xlabel(label_prefix+' %d' % (i + 1))


def renderSourceImage(source_image, firstTextureLocations, secondTextureLocations):
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(source_image, cmap=plt.cm.gray, vmin=0, vmax=255)

    for (y, x) in firstTextureLocations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')

    for (y, x) in secondTextureLocations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')

    ax.set_xlabel('Source image')


def renderGlcm(firstTexturePatches, secondTexturePatches):
    xs = []
    ys = []

    for patch in (firstTexturePatches + secondTexturePatches):
        glcm = greycomatrix(patch, distances=[5], angles=[
            0], levels=256, symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(firstTexturePatches)], ys[:len(
        firstTexturePatches)], 'go', label='Land')
    ax.plot(xs[len(secondTexturePatches):],
            ys[len(secondTexturePatches):], 'bo', label='Sea')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()


land_locations = [(1374,  801), (1580,  187), (1050,  576), (2414,  101)]
sea_locations = [(1457,  2557), (925,  2077), (413,  2573), (3241,  3409)]

image = np.array(Image.open('./assets/nasa_1.jpg').convert('L'))
fig = plt.figure(figsize=(8, 8))

land_patches = createPatches(land_locations)
sea_patches = createPatches(sea_locations)

renderSourceImage(image, land_locations, sea_locations)
renderGlcm(land_patches, sea_patches)
renderPatches(land_patches, "Land", 1)
renderPatches(sea_patches, "Sea", 2)

plt.tight_layout()
plt.show()
