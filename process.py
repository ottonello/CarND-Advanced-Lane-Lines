import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import util

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# file = 'straight_lines1.jpg'
# Perform thresholding before perspective transformation
thresh = True
file = 'test5.jpg'
source_files = 'test_images'
output_files = 'output_images'
base_filename = os.path.splitext(file)[0]

# Perspective warping source and destination points
src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])

print(base_filename)

# Read in an image
img = cv2.imread(os.path.join(source_files, file))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mpimg.imsave(os.path.join(output_files, base_filename + "_1_orig.jpg"), img)

img = util.cal_undistort(img, mtx, dist)

mpimg.imsave(os.path.join(output_files, base_filename + "_2_undistorted.jpg"), img)

if thresh:
    img = util.threshold(img)

    mpimg.imsave(os.path.join(output_files, base_filename + "_3_thresholded.jpg"), img, cmap='gray')
    img = util.perspective_transform(img, src, dst)

    mpimg.imsave(os.path.join(output_files, base_filename + "_4_perspective.jpg"), img, cmap='gray')

    histogram = util.get_histogram(img)
    plt.plot(histogram)
    plt.savefig(os.path.join(output_files, base_filename + "_5_histogram.jpg"))
else:
    img = util.perspective_transform(img, src, dst)

    mpimg.imsave(os.path.join(output_files, base_filename + "_4_perspective.jpg"), img)
