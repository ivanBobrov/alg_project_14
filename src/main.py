import skimage.color as color
import matplotlib.pyplot as plt
import cv2
import numpy as np
import Cluster_Ensembles as ce
import math

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

test_image = cv2.imread("resources/mondrian.jpg")
image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
img = img_as_float(image_rgb)


def calculate_segmentations(image):
    return [#felzenszwalb(image, scale=1000, sigma=2, min_size=30),
            #felzenszwalb(image, scale=2000, sigma=1.8, min_size=30),
            #felzenszwalb(image, scale=1500, sigma=1.5, min_size=30),
            #felzenszwalb(image, scale=900, sigma=1.9, min_size=30),
            slic(image, n_segments=30, compactness=15, max_iter=100, min_size_factor=0.1),
            slic(image, n_segments=30, compactness=10, max_iter=100, min_size_factor=0.2),
            slic(image, n_segments=30, compactness=100, max_iter=100, min_size_factor=0.15),
            slic(image_hsv, n_segments=30, compactness=75, max_iter=100, min_size_factor=0.01),
            quickshift(image, kernel_size=10, max_dist=100, ratio=0.1),
            #felzenszwalb(image, scale=2000, sigma=2, min_size=30),
            #felzenszwalb(image, scale=2000, sigma=2, min_size=30),
            #felzenszwalb(image, scale=1900, sigma=1.9, min_size=30),
            watershed(sobel(rgb2gray(image)), markers=30, compactness=0.0005, connectivity=.18),
            watershed(sobel(rgb2gray(image)), markers=30, compactness=0.001, connectivity=0.5),
            watershed(sobel(rgb2gray(image)), markers=30, compactness=0.0007, connectivity=1),
            watershed(sobel(rgb2gray(image)), markers=30, compactness=0.0009, connectivity=1.1)
            ]


def draw_segmentations(segmentations, image):
    n = math.ceil(math.sqrt(len(segmentations)))
    plt.figure(dpi=200 * n)
    fig, ax = plt.subplots(n, n, figsize=(10, 10), sharex=True, sharey=True)
    for i in range(n):
        for j in range(n):
            if i + n * j >= len(segmentations):
                break
            segment = segmentations[i + n * j]
            ax[j, i].imshow(mark_boundaries(image, segment))

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


print("Calculating segmentations")
segmentations = calculate_segmentations(img)

print("Combining clusterings")
cluster_runs = np.asarray(list(map(lambda s: s.flatten(), segmentations)))
consensus = ce.cluster_ensembles(cluster_runs, verbose=True, N_clusters_max=30)
consensus = consensus.reshape(segmentations[0].shape)

print("Drawing results")
draw_segmentations(segmentations, img)

plt.figure(dpi=300)
plt.imshow(mark_boundaries(img, consensus))
plt.axis('off')
plt.show()
