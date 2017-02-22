##Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Original chessboard"
[image1]: ./output_images/calibrated1.jpg "Corrected chessboard"
[image2]: ./output_images/test1.jpg_1_orig.jpg "Original road"
[image3]: ./output_images/test1.jpg_2_undistorted.jpg "Road Transformed"
[image4]: ./output_images/test1.jpg_3_thresholded.jpg "Binary Example"
[image5]: ./output_images/straight_lines1_undistorted_ptransf1.jpg "Warp Example"
[image6]: ./output_images/test1.jpg_4_perspective.jpg "Warp Example"
[image7]: ./output_images/test1.jpg_5_histogram.jpg "Histogram"
[image8]: ./output_images/test1.jpg_windows.jpg "Detected windows"
[image9]: ./output_images/test1.jpg_7_detected_lane.jpg "Detected lane plot"
[image10]: ./output_images/test1.jpg_8_output.jpg "Output"
[video1]: ./solution_video.mp4 "Video"

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fiel `calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
 calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a 
 copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with
  the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients
 using the `cv2.calibrateCamera()` function.
   This was done for all the images in the `camera_cal` folder.  
 
 I applied this distortion correction to the test image using the
  `cv2.undistort()` function and obtained this result: 

Original: 

![alt text][image0]

Corrected:

![alt text][image1]

###Pipeline (single images)

####1. Distortion correction
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

The pipeline takes two arguments for camera distortion correction. These two arguments are called:
- *mtx*: The camera matrix(as returned, i.e. from cv2.calibrateCamera)
- *dist*: The camera distortion coefficient(as returned, i.e. from cv2.calibrateCamera)

These two arguments are fed into the `cv2.undistort` function(lines #23-24, `pipeline.py`). This call returns the resulting,
undistorted image:

![alt text][image3]

####2. Thresholding

Gradient detection using Sobel filters, as well as a color space transformation into
 HLS is used for generating a binary thresholded image with candidate pixels for detected lines.

The auxiliary functions and the combining method are located between lines 27 and 83 in `pipeline.py`.

I used the following thresholding mechanisms for the line detection:
- Absolute value of the gradient in the x/y directions
- Gradient direction
- Gradient magnitude
- 'S' channel threshold after conversion to HLS color space. 

Convenient thresholding values for each of these mechanisms were empirically determined.

An output sample looks like thissss:
s
![alt text][image4]

####3. Perspective transform


The code for my perspective transform includes a function called `perspective_transform()`, 
which appears in lines 86 through 90 in the file `pipeline.py`.  
The `transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  

I found the source points in the image by eyeballing on a straight line image with the final points in the perspective
down the road and the source ones in the bottom of the picture. I also made sure the transformation preserved the
distances from left and right and that the aspect ratio is conserved. This makes calculations easier later.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test 
image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

Warped sample:

![alt text][image6]

####4. Lane detection and polynomial fitting

The first step for finding the lines was to get a histogram of the frequency of pixels in the lower half of the image.
This is used for finding the candidate x,y points where we will assume the center of the lines will be located.
The histogram is obtained in lines 93 to 95 in `pipeline.py`, and if we plot the results we will get something
like the following plot:

![alt text][image7]

After this, a sliding window search is performed in the left and right halves of the image. This is done
starting from the lower position, where the line is initially detected from the histogram, then
by progressively moving upwards and repositioning the window center towards the mean of the x-positions 
for all the non-zero pixels within the current window.
This is how these windows look when plotted and proyected into the original image:

![alt text][image8]

The positions of all pixels for each window are all added up into a single array. From all the positions of these pixels,
which hopefully will describe approximately the positions of the lines, we can perform a polynomial fitting using 
the `np.polyfit()` function, which takes the `y` and `x` coordinates of a set of points and returns an array
 of coefficients for a polynomial fit of the requested degree.
 
 In this case, as appropriate we use a polynomial of second degree. The curvature of the lane lines should rarely require
 a higher degree. The three resulting coefficients are then used to plot the lines.
 
 The code for the sliding window detection and polynomial regression is contained in the `find_lane` function 
 in the `pipeline.py` file(lines 98-167)

####5. Calculating radius of curvature and lane position

Once the fitting parameters for the curves are found, the radius can be found from the function by using the formula as given
in http://www.intmath.com/applications-differentiation/8-radius-curvature.php.  
As  given, this method will return the radius in pixel units of the curve. To convert this into meters we first convert
these parameters to the physical measurements by multiplying both x and y measurements by the ratio of meters/pixel we
found by measuring the lane lines and taking into account the line width and an approximated value of 30m for the length
of the considered area. These two values are constants and its calculation can be found in lines 11-13 of the `pipeline.py`
 file. The radius calculation itself is located in the `get_curverad` function in the same file, lines 185-199.
   

####6. Plotting the lane back into the image

I implemented this step in lines 301 through 309, in my code in `pipeline.py` in the function `map_lane()`.
This function only draws the lines into an empty image, with the results being as follows:

![alt text][image9]

After performing the inverse perspective transformation and overlying it on the original image (done in lines 283 and 284),
The result looks like follows:

![alt text][image10]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./solution_video.mp4)

Alternative: [youtube](https://www.youtube.com/watch?v=ur_vsFF68kA&feature=youtu.be)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

I think the main problem in this project is about finding the correct thresholds and combination of filtering tecniques
 so that lines are properly detected while minimizing the amount of noise and residual borders detected.  
The difficulty in this task is especially noticeable in road sections where there are adjacent, parallel lines to the lane,
 like where there are road dividers which project a shadow over the pavement. In other places, there are shadows
 that are transverse to the road lines, these shadows work add noise especially in the 'S' channel, which I found to
 be very useful to detect the lines. There is possibly a combination of channels capable of eliminating this noise, but I was unable
 to find one so far.