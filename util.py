import cv2
import numpy as np

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs = np.absolute(sobel)
    scaled = np.uint8(abs/np.max(abs) * 255)
    thresholded = np.zeros_like(scaled)
    thresholded[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return thresholded

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag = np.sqrt(np.power(gradx,2) + np.power(grady,2))
    scaled = np.uint8(255 * mag/np.max(mag))
    mask = np.zeros_like(scaled)
    mask[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_x = np.abs(gradx)
    abs_y = np.abs(grady)
    directions = np.arctan2(abs_y, abs_x)
    binary_output = np.zeros_like(directions)
    binary_output[(directions >= thresh[0]) & (directions <= thresh[1])] = 1
    return binary_output


def s_channel_threshold(image, thresh=(90, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def threshold(image):
    ksize=3
    s_thresh = s_channel_threshold(image, thresh=(170, 255))

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(40, 120))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(40, 120))
    dir_binary = dir_threshold(image,  sobel_kernel=15, thresh=(0.7, 1.2))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(50, 200))
    combined = np.zeros_like(dir_binary)

    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | s_thresh ==1] = 1

    return combined


def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def get_histogram(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    return histogram
