import cv2
import numpy as np
#from matplotlib import pyplot as plt

cap = cv2.VideoCapture('video.avi')
# Trained XML classifiers describes some features of some object we want to detect
cars = cv2.CascadeClassifier('cars.xml')

# PARAMETERS FOR VEHICLE COUNTING
# Vehicles travelling up the screen
count_up = 0
# Vehicles travelling down the screen
count_down = 0
# Get dimensions of frame
w_frame = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h_frame = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Set line to be vertically centered on frame
line = int(h_frame/2)
# Set line colour to be blue
line_color = (255, 0,  0)
# Create points for line [x,y]
p1 = [0, line]
p2 = [w_frame, line]
# Assign line position
line_pos= np.array([p1,p2], np.int32)


# Background Subtraction Using Weighted Averages
_, frame = cap.read()
# Create numpy array of frames
avg = np.float32(frame)
while True:
    #capture frame by frame
    _, frame = cap.read()
    if frame is None:
        break
    #Get average of frames
    cv2.accumulateWeighted(frame, avg, 0.01)
    #performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type
    a = cv2.convertScaleAbs(avg)
    avg_frame = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Subtract background from foreground
    diff = cv2.absdiff(avg_frame, gray_frame)
    #     # Set a threshold to remove gray noise
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#     # Remove a good portion of noise by opening
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     # Create structuring element for closing
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
#     # Close gaps using closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)

    # BLOB TRACKING
    # Set parameters for keypoints (blob tracking)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 10
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 30000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1
    # Create blob detector with set parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs in foreground
    track = detector.detect(closing)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # (drawKeypoints(frames, keypoints, output image, colour, flags)
    blob_tracking = cv2.drawKeypoints(closing, track, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # ADD CENTERED LINE TO FRAME FOR VEHICLE COUNTING
    frame = cv2.polylines(frame, [line_pos], False, line_color, thickness=1)


    #Display background
    cv2.imshow('Frame', frame)
    cv2.imshow('avg', a)
    cv2.imshow('diff', diff)
    cv2.imshow('Thres', thresh)
    cv2.imshow('Closing', closing)
    cv2.imshow('Blob', blob_tracking)
    #displays window for 20 ms
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()


print("allo")
