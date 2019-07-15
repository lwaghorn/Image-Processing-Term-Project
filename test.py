import cv2
import numpy as np

cap = cv2.VideoCapture('video.avi')
# Trained XML classifiers describes some features of some object we want to detect
# cars = cv2.CascadeClassifier('cars.xml')

# Background extraction using Mixture of Gaussians
bsub = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read frames
    _, frame = cap.read()
    if frame is None:
        break

    # Apply background subtraction
    fg = bsub.apply(frame)
    # Create structuring element for opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # Remove a good portion of noise by opening
    opening = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    # Create structuring element for closing
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # Close gaps using closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    # Set a threshold to remove gray noise
    (thresh, thres) = cv2.threshold(closing, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Set parameters for keypoints (blob tracking)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 30
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 300000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1
    # Create blob detector with set parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs in foreground
    keypoints = detector.detect(thres)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # (drawKeypoints(frames, keypoints, output image, colour, flags)
    im_with_keypoints = cv2.drawKeypoints(thres, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show original video
    cv2.imshow('Frame', frame)
    # Show foreground
    cv2.imshow('Foreground', fg)
    # Show result of opening
    cv2.imshow('Opening', opening)
    # Show result of closing
    cv2.imshow('Closing', closing)
    # Show threshold foreground
    cv2.imshow('Thres', thres)
    # Show keypoints
    cv2.imshow('Keypoints', im_with_keypoints)
    # Speed of frames
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




# # Background Subtraction Using Weighted Averages
# _, frame = cap.read()
# # Create numpy array of frames
# avg = np.float32(frame)
# while True:
#     #capture frame by frame
#     _, frame = cap.read()
#     if frame is None:
#         break
#     #Get average of frames
#     cv2.accumulateWeighted(frame, avg, 0.01)
#     #performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type
#     a = cv2.convertScaleAbs(avg)
#     avg_frame = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Subtract background from foreground
#     diff = cv2.absdiff(avg_frame, gray_frame)
#     #Display background
#     cv2.imshow('avg', a)
#     cv2.imshow('diff', diff)
#     #displays window for 20 ms
#     k = cv2.waitKey(20)
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# cap.release()


print("allo")
