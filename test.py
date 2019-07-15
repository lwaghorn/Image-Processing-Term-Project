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


# Background extraction using Mixture of Gaussians
bsub = cv2.createBackgroundSubtractorMOG2()

#hog = cv2.HOGDescriptor()

while True:
    # Read frames
    _, frame = cap.read()
    if frame is None:
        break

    # Working on histogram
    # color = ('b','g','r')
    # for i,col in enumerate(color):
    #     # calcHist([image], [0 for gray scale, 0 1 or 2 for colored], None for mask, histSize for full scale [256], ranges [0,256]
    #     histr = cv2.calcHist([frame],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.show()

    # BACKGROUND SUBTRACTION
    fg = bsub.apply(frame)
    # Create structuring element for opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # Remove a good portion of noise by opening
    opening = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    # Create structuring element for closing
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    # Close gaps using closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    # Set a threshold to remove gray noise
    _, thresh = cv2.threshold(closing, 220, 255, cv2.THRESH_BINARY)

    # BLOB TRACKING
    # Set parameters for keypoints (blob tracking)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 20
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 3000000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1
    # Create blob detector with set parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs in foreground
    track = detector.detect(thresh)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # (drawKeypoints(frames, keypoints, output image, colour, flags)
    blob_tracking = cv2.drawKeypoints(thresh, track, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # ADD CENTERED LINE TO FRAME FOR VEHICLE COUNTING
    frame = cv2.polylines(frame, [line_pos], False, line_color, thickness=1)


    # SHOW ALL IMAGES
    # Show original video
    cv2.imshow('Frame', frame)
    # Show foreground
    cv2.imshow('Foreground', fg)
    # Show result of opening
    cv2.imshow('Opening', opening)
    # Show result of closing
    cv2.imshow('Closing', closing)
    # Show threshold foreground
    cv2.imshow('Threshold', thresh)
    # Show blob tracking
    cv2.imshow('Tracking', blob_tracking)
    #cv2.imshow('hist', hist)
    # Speed of frames
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




