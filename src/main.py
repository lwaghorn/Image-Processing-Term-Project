import cv2
import numpy as np

crossed_up = 0
crossed_down = 0

crossed_down_color = (255, 0, 0)
crossed_up_color = (255, 0, 255)

font = cv2.FONT_HERSHEY_SIMPLEX
objects = []
object_frame_timeout = 5
object_id = 1


def update_display(new_frame):
    str_up = 'UP: ' + str(crossed_up)
    str_down = 'DOWN: ' + str(crossed_down)
    new_frame = cv2.polylines(new_frame, [pts_L1], False, crossed_down_color, thickness=2)
    new_frame = cv2.polylines(new_frame, [pts_L2], False, crossed_up_color, thickness=2)
    new_frame = cv2.polylines(new_frame, [pts_L3], False, (255, 255, 255), thickness=1)
    new_frame = cv2.polylines(new_frame, [pts_L4], False, (255, 255, 255), thickness=1)
    cv2.putText(new_frame, str_up, (10, 40), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(new_frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(new_frame, str_down, (10, 90), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(new_frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', new_frame)


if __name__ == "__main__":

    cap = cv2.VideoCapture("compressed2.avi")

    # Get width and height of video

    video_width = cap.get(3)
    video_height = cap.get(4)
    video_area = video_height * video_width
    areaTH = video_area / 400

    # Lines
    line_up = int(4 * (video_height / 10))
    line_down = line_up

    up_limit = int(1 * (video_height / 10))
    down_limit = int(9 * (video_height / 10))

    pt1 = [0, line_down]
    pt2 = [video_width, line_down]
    pts_L1 = np.array([pt1, pt2], np.int32)
    pts_L1 = pts_L1.reshape((-1, 1, 2))
    pt3 = [0, line_up]
    pt4 = [video_width, line_up]
    pts_L2 = np.array([pt3, pt4], np.int32)
    pts_L2 = pts_L2.reshape((-1, 1, 2))

    pt5 = [0, up_limit]
    pt6 = [video_width, up_limit]
    pts_L3 = np.array([pt5, pt6], np.int32)
    pts_L3 = pts_L3.reshape((-1, 1, 2))
    pt7 = [0, down_limit]
    pt8 = [video_width, down_limit]
    pts_L4 = np.array([pt7, pt8], np.int32)
    pts_L4 = pts_L4.reshape((-1, 1, 2))

    # Background Subtract
    background_subtract = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Kernals
    kernelOp = np.ones((5, 5), np.uint8)
    kernelOp2 = np.ones((21, 21), np.uint8)

    while cap.isOpened():
        has_frame, frame = cap.read()
        if has_frame:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            foreground_mask = background_subtract.apply(frame)
            # cv2.imshow('Subtracted', foreground_mask)

            # Binary Image
            has_frame, imBin = cv2.threshold(foreground_mask, 150, 255, cv2.THRESH_BINARY)
            # cv2.imshow('bin', imBin)

            # Erode with small window to remove noise, then large dilation to join hood of car to roof
            morphological_mask = cv2.morphologyEx(imBin, cv2.MORPH_ERODE, kernelOp)
            morphological_mask = cv2.morphologyEx(morphological_mask, cv2.MORPH_DILATE, kernelOp2)

            cv2.imshow('Morphed Image', morphological_mask)

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

                    car = next((car for car in objects if (car.in_contour(centroid_x, centroid_y, contour_width,
                                                                          contour_height) and not car.is_done())), None)
                    if car is not None:
                        car.update_coordinates(centroid_x, centroid_y)
                        cv2.putText(frame, str(car.get_id()), (car.x + 5, car.y), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                        if car.going_up(line_up):
                            crossed_up += 1
                            print("ID:", car.get_id(), "was the ", crossed_up, 'crossing up')
                        elif car.going_down(line_down):
                            crossed_down += 1
                            print("ID:", car.get_id(), "was the ", crossed_down, 'crossing down')
                    else:  # New Car
                        objects.append(objects.Object(object_id, centroid_x, centroid_y, object_frame_timeout))
                        object_id += 1

                    cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    img = cv2.rectangle(frame, (contour_x, contour_y), (contour_x + contour_width, contour_y + contour_height), (0, 255, 0), 2)

            update_display(frame)

            for car in objects:
                car.age()
                car.check_health()

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
