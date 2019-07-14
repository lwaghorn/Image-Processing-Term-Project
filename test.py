import cv2
import numpy as np

cap = cv2.VideoCapture('video.avi')
# Trained XML classifiers describes some features of some object we want to detect
# cars = cv2.CascadeClassifier('cars.xml')

#Background extraction using Mixture of Gaussians
bsub = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read frames
    _, frame = cap.read()
    if frame is None:
        break
    # Apply background subtraction
    fg = bsub.apply(frame)
    # Show frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground', fg)
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
