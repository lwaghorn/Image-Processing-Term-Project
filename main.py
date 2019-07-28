import time
import cv2
import numpy as np
import vehicles

cnt_up = 0
cnt_down = 0

cap = cv2.VideoCapture("video.avi")

# Get width and height of video

contour_width = cap.get(3)
contour_height = cap.get(4)
frameArea = contour_height * contour_width
areaTH = frameArea / 400

# Lines
line_up = int(2 * (contour_height / 5))
line_down = int(3 * (contour_height / 5))

up_limit = int(1 * (contour_height / 5))
down_limit = int(4 * (contour_height / 5))

print("Red line y:", str(line_down))
print("Blue line y:", str(line_up))

line_down_color = (255, 0, 0)
line_up_color = (255, 0, 255)

pt1 = [0, line_down]
pt2 = [contour_width, line_down]
pts_L1 = np.array([pt1, pt2], np.int32)
pts_L1 = pts_L1.reshape((-1, 1, 2))
pt3 = [0, line_up]
pt4 = [contour_width, line_up]
pts_L2 = np.array([pt3, pt4], np.int32)
pts_L2 = pts_L2.reshape((-1, 1, 2))

pt5 = [0, up_limit]
pt6 = [contour_width, up_limit]
pts_L3 = np.array([pt5, pt6], np.int32)
pts_L3 = pts_L3.reshape((-1, 1, 2))
pt7 = [0, down_limit]
pt8 = [contour_width, down_limit]
pts_L4 = np.array([pt7, pt8], np.int32)
pts_L4 = pts_L4.reshape((-1, 1, 2))

# Background Subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Kernals
kernalOp = np.ones((3, 3), np.uint8)
kernalOp2 = np.ones((5, 5), np.uint8)
kernalCl = np.ones((11, 11), np.uint)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1

while (cap.isOpened()):
    has_frame, frame = cap.read()
    foreground_mask = background_subtractor.apply(frame)
    foreground_mask_2 = background_subtractor.apply(frame)

    cv2.imshow('Subtracted', foreground_mask_2)

    if has_frame:

        # Binarization
        has_frame, imBin = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
        has_frame, imBin2 = cv2.threshold(foreground_mask_2, 200, 255, cv2.THRESH_BINARY)

        cv2.imshow('bin', imBin2)

        # Opening i.e First Erode the dilate
        morphological_mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
        morphological_mask_2 = cv2.morphologyEx(imBin2, cv2.MORPH_CLOSE, kernalOp)

        cv2.imshow('dilate', morphological_mask)

        # Closing i.e First Dilate then Erode
        morphological_mask = cv2.morphologyEx(morphological_mask, cv2.MORPH_CLOSE, kernalCl)
        morphological_mask_2 = cv2.morphologyEx(morphological_mask_2, cv2.MORPH_CLOSE, kernalCl)

        cv2.imshow('erode', morphological_mask)

        # Find Contours
        contours, _ = cv2.findContours(morphological_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > areaTH:
                # Tracking
                contour_moments = cv2.moments(contour)

                # Centroid
                centroid_x = int(contour_moments['m10'] / contour_moments['m00'])
                centroid_y = int(contour_moments['m01'] / contour_moments['m00'])

                if centroid_y not in range(up_limit, down_limit):
                    continue

                contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)

                car = next((car for car in cars if (car.in_contour(contour_x, contour_y, contour_width, contour_height)) and not car.is_done()), None)
                if car is not None:
                    car.update_coordinates(centroid_x, centroid_y)
                    if car.going_up(line_up):
                        cnt_up += 1
                        print("ID:", car.get_id(), 'crossed going up at', time.strftime("%c"))
                    elif car.going_down(line_down):
                        cnt_down += 1
                        print("ID:", car.get_id(), 'crossed going up at', time.strftime("%c"))
                else:  # New Car
                    cars.append(vehicles.Car(pid, centroid_x, centroid_y, max_p_age))
                    pid += 1

                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (contour_x, contour_y), (contour_x + contour_width, contour_y + contour_height), (0, 255, 0), 2)

        str_up = 'UP: ' + str(cnt_up)
        str_down = 'DOWN: ' + str(cnt_down)
        frame = cv2.polylines(frame, [pts_L1], False, line_down_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L2], False, line_up_color, thickness=2)
        frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
