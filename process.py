import cv2
import matplotlib.image as mpimg
import numpy as np
import os
import pickle

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# file = 'straight_lines1.jpg'
file = 'test5.jpg'
source_files = 'test_images'
output_files = 'output_images'
base_filename = os.path.splitext(file)[0]
print(base_filename)

# Read in an image
img = cv2.imread(os.path.join(source_files, file))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


def h_channel_threshold(image, thresh=(90, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def threshold(image, h_thresh=(90, 255)):
    return h_channel_threshold(image, h_thresh)


def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


mpimg.imsave(os.path.join(output_files, base_filename + "_1_orig.jpg"), img)

img = cal_undistort(img, mtx, dist)

mpimg.imsave(os.path.join(output_files, base_filename + "_2_undistorted.jpg"), img)

img = threshold(img, (160, 250))

mpimg.imsave(os.path.join(output_files, base_filename + "_3_thresholded.jpg"), img, cmap='gray')
#//564,472 717
src = np.float32([[564,472], [719,472], [269,676],[1034,676]])
dst = np.float32([[200,0],   [1080,0],  [200,720],[1080,720]])
img = perspective_transform(img, src, dst)

mpimg.imsave(os.path.join(output_files, base_filename + "_4_perspective.jpg"), img, cmap='gray')
# mpimg.imsave(os.path.join(output_files, base_filename + "_4_perspective.jpg"), img)
