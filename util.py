import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import collections


def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs = np.absolute(sobel)
    scaled = np.uint8(abs / np.max(abs) * 255)
    thresholded = np.zeros_like(scaled)
    thresholded[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return thresholded


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag = np.sqrt(np.power(gradx, 2) + np.power(grady, 2))
    scaled = np.uint8(255 * mag / np.max(mag))
    mask = np.zeros_like(scaled)
    mask[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_x = np.abs(gradx)
    abs_y = np.abs(grady)
    directions = np.arctan2(abs_y, abs_x)
    binary_output = np.zeros_like(directions)
    binary_output[(directions >= thresh[0]) & (directions <= thresh[1])] = 1
    return binary_output


def hls_s_channel_threshold(image, thresh=(90, 255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hsv[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def threshold(image):
    ksize = 3
    s_thresh = hls_s_channel_threshold(image, thresh=(100, 255))

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(15, 210))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(15, 210))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.2))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(50, 200))
    combined = np.zeros_like(dir_binary)

    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((s_thresh==1))] = 1

    return combined


def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def get_histogram(img):
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    return histogram


def find_lane(img, histogram, left_fit=None, right_fit=None):
    if left_fit is None and right_fit is None:
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        # 720 / 9 = 80
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    else:
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


# Takes RGB image
def pipeline(orig, mtx, dist, src, dst, base_filename,
             prev_lfit=None, prev_rfit=None, l_acc=None, r_acc=None,
             output_files='output_images', debug=False):
    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_1_orig.jpg"), orig)

    img = cal_undistort(orig, mtx, dist)

    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_2_undistorted.jpg"), img)

    img = threshold(img)

    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_3_thresholded.jpg"), img, cmap='gray')

    img = perspective_transform(img, src, dst)

    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_4_perspective.jpg"), img, cmap='gray')

    histogram = get_histogram(img)

    if debug:
        plt.plot(histogram)
        plt.savefig(os.path.join(output_files, base_filename + "_5_histogram.jpg"))
        plt.close()

    lfit, rfit = find_lane(img, histogram, left_fit=prev_lfit, right_fit=prev_rfit)

    if l_acc == None:
        l_acc = collections.deque(maxlen=10)
    if r_acc == None:
        r_acc = collections.deque(maxlen=10)

    l_acc.append(lfit)
    r_acc.append(rfit)

    # print(np.shape(lfit))
    # print(np.shape(r_acc))
    # print(np.shape(np.sum(l_acc, 0)))
    # print(l_acc)
    lfit = np.array(np.sum(l_acc, 0)) / len(l_acc)
    rfit = np.array(np.sum(r_acc, 0)) / len(r_acc)
    # print(lfit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = lfit[0] * ploty ** 2 + lfit[1] * ploty + lfit[2]
    right_fitx = rfit[0] * ploty ** 2 + rfit[1] * ploty + rfit[2]

    # print(np.shape(ploty))
    l_points = np.squeeze(np.array(np.dstack((left_fitx, ploty)), dtype='int32'))
    r_points = np.squeeze(np.array(np.dstack((right_fitx, ploty)), dtype='int32'))

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    out_img = np.zeros_like(orig)
    points_rect = np.concatenate((r_points, l_points[::-1]), 0)
    cv2.fillPoly(out_img, [points_rect], (0, 255, 0))
    cv2.polylines(out_img, [l_points], False, (255, 0, 0), 15)
    cv2.polylines(out_img, [r_points], False, (0, 0, 255), 15)

    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_7_detected_lane.jpg"), out_img, cmap='gray')

    # Draw lane into original image
    out_img = perspective_transform(out_img, dst, src)
    out_img = cv2.addWeighted(orig, .5, out_img, .5, 0.0, dtype=0)
    cv2.putText(out_img, "Radius: %.2fm" % ((left_curverad + right_curverad) / 2), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0))

    if debug:
        mpimg.imsave(os.path.join(output_files, base_filename + "_8_output.jpg"), out_img, cmap='gray')

    return out_img, lfit, rfit, l_acc, r_acc
