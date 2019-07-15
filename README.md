Image Processing Term Project
Liam Waghorn & Amy Barrett

test.py
Note: Currently there are two methods of background subtraction.
1) Mixture of Gaussians
    The MOG function in opencv is used to subtract the background of the frames. This results in a video of the moving
    vehicles in the foreground. More noisy.
2) Weighted Average
    The average of all frames is calculated to produce the background. The background is then subtracted from the
    foreground to produce a video of the moving vehicles. Less noisy but less clear.

video.avi
Random traffic video found on the internet

cars.xml
Random trained vehicle classifier (not used yet)


Useful links:
https://www.youtube.com/watch?v=inCUJ0JM5ng
Source code for video above: https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
https://github.com/amartya-k/vision/blob/master/main.py
Easy implementation: https://www.geeksforgeeks.org/opencv-python-program-vehicle-detection-video-frame/
