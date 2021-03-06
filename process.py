import cv2
import numpy as np
import os
import pickle
import pipeline
import glob

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# file = 'straight_lines1.jpg'
# Perform thresholding before perspective transformation
thresh = True
file = 'test2/167.png'
source_files = 'test_images'

# Perspective warping source and destination points
src = np.float32([[600,451], [680,451], [243,720],[1057,720]])
dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])

for file in glob.glob(os.path.join(source_files, "*.jpg")):
    base_filename = os.path.basename(file)
    print(base_filename)

    # Read in an image
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pipeline.pipeline(img, mtx, dist, src, dst, base_filename, output_files="output_images", debug=True)

