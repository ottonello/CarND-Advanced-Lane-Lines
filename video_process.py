from moviepy.editor import VideoFileClip
import pickle
import util
import numpy as np

# Perspective warping source and destination points
src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("calibration.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def process_image(image):
    return util.pipeline(image, mtx, dist, src, dst, "", debug=False)

output_video = "solution_video.mp4"
clip1 = VideoFileClip("project_video.mp4")
output_clip= clip1.fl_image(process_image)
output_clip.write_videofile(output_video, audio=False)