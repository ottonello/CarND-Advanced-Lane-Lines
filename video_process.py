from moviepy.editor import VideoFileClip
import pickle
import pipeline
import numpy as np
import collections

# Perspective warping source and destination points
src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Take average of this number of frames
AVERAGE_FRAMES = 10

prev_lfit=None
prev_rfit=None
l_acc = collections.deque(maxlen=AVERAGE_FRAMES)
r_acc = collections.deque(maxlen=AVERAGE_FRAMES)

def process_image(image):
    global prev_rfit, prev_lfit, l_acc, r_acc
    img, prev_lfit, prev_rfit, l_acc, r_acc = pipeline.pipeline(image, mtx, dist, src, dst, "", prev_lfit=prev_lfit, prev_rfit=prev_rfit, l_acc=l_acc, r_acc=r_acc, debug=False)
    return img

output_video = "solution_video.mp4"
clip1 = VideoFileClip("project_video.mp4")
output_clip= clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)