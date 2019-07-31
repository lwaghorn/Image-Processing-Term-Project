import cv2
import numpy as np
from reasources.objects_edited import Object

# Variables to count cars going up and down screen
crossed_up = 0
crossed_down = 0

# Assigning colors to items printed on screen
crossing_line_color = (204, 50, 153)
crossed_down_color = (255, 0, 0)
crossed_up_color = (0, 0, 255)
centroid_color = (0, 140, 255)
rectangle_color = (0, 255, 0)


font = cv2.FONT_HERSHEY_SIMPLEX
tracking_objects = []
# Setting max amount frames for object to not move
object_frame_timeout = 5
# Initalizing car id
object_id = 1

# Outputting lines and text onto frame
def update_display(new_frame):
    new_frame = cv2.line(new_frame,(0,int(line_down)),(int(video_width),int(line_down)),crossing_line_color,2)
    new_frame = cv2.line(new_frame,(0,int(line_up)),(int(video_width),int(line_up)),crossing_line_color,2)
    new_frame = cv2.line(new_frame,(0,int(up_limit)),(int(video_width),int(up_limit)),(0,0,0),1)
    new_frame = cv2.line(new_frame,(0,int(down_limit)),(int(video_width),int(down_limit)),(0,0,0),1)
    cv2.putText(new_frame, 'Up:'  +  str(crossed_up), (550, 450), font, 2, crossed_up_color, 4, cv2.LINE_AA)
    cv2.putText(new_frame, 'Down:'  +  str(crossed_down), (250, 450), font, 2, crossed_down_color, 4, cv2.LINE_AA)
    cv2.imshow('Frame', new_frame)


if __name__ == "__main__":

    # Read video
    cap = cv2.VideoCapture("videos/compressed2.avi")

    # Get width, height are area of video
    video_width = cap.get(3)
    video_height = cap.get(4)
    video_area = video_height * video_width
    area_threshold = video_area / 400

    # Lines
    # Line to detect cars going up the screen
    line_up = int(4 * (video_height / 10))
    # Line to detect cars going down the screen
    line_down = line_up
    # Upper detection limit line
    up_limit = int(1 * (video_height / 10))
    # Lower detection limit line
    down_limit = int(9 * (video_height / 10))

    # Background Subtract
    background_subtract = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    # Kernals used for morophological masking
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((21, 21), np.uint8)

    while cap.isOpened():
        has_frame, frame = cap.read()
        if has_frame:
            # Rotate image because it originally loads upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Apply background subtraction to isolate foreground
            foreground_mask = background_subtract.apply(frame)
            # cv2.imshow('Subtracted', foreground_mask)

            # Binary image. Thresholding to produce black and white image
            has_frame, imBin = cv2.threshold(foreground_mask, 150, 255, cv2.THRESH_BINARY)
            # cv2.imshow('bin', imBin)

            # Remove S&P noise with median blur
            morphological_mask = cv2.medianBlur(imBin, 5)
            # cv2.imshow('median_blur', morphological_mask)

            # Erode with small window to remove noise
            morphological_mask = cv2.morphologyEx(imBin, cv2.MORPH_ERODE, kernel_erode)
            cv2.imshow('Eroded Image', morphological_mask)

            # Dilate with larger window to join hood of car to roof
            morphological_mask = cv2.morphologyEx(morphological_mask, cv2.MORPH_DILATE, kernel_dilate)
            cv2.imshow('Morphed Image', morphological_mask)

            # Find contours (outlines) of each vehicle
            contours, _ = cv2.findContours(morphological_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_threshold:
                    # Calculate moments of contour
                    contour_moments = cv2.moments(contour)

                    # Create centroid using moments
                    centroid_x = int(contour_moments['m10'] / contour_moments['m00'])
                    centroid_y = int(contour_moments['m01'] / contour_moments['m00'])

                    # Check if centroid is within upper and lower limits
                    if centroid_y not in range(up_limit, down_limit):
                        continue
                    # Draw approximate rectangle around image
                    contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)

                    # Track cars only if car is not done and
                    car = next((car for car in tracking_objects if (car.in_contour(centroid_x, centroid_y, contour_width,
                                                                          contour_height) and not car.is_done())), None)
                    if car is not None:
                        # Update coordinates of car if in frame and moving
                        car.update_coordinates(centroid_x, centroid_y)
                        # If car passes up line, increase counter
                        if car.going_up(line_up):
                            crossed_up += 1
                            print("ID:", car.get_id(), "was the ", crossed_up, 'crossing up')
                        # If car passes down line, increase counter
                        elif car.going_down(line_down):
                            crossed_down += 1
                            print("ID:", car.get_id(), "was the ", crossed_down, 'crossing down')
                    # If car is entering frame (new) create object ID and track
                    else:
                        tracking_objects.append(Object(object_id, centroid_x, centroid_y, object_frame_timeout))
                        object_id += 1

                    # Create centroid centered on vehicle to track if it passed line
                    cv2.circle(frame, (centroid_x, centroid_y), 5, centroid_color, -1)
                    # Create rectangle around each vehicle
                    img = cv2.rectangle(frame, (contour_x, contour_y), (contour_x + contour_width, contour_y + contour_height), rectangle_color, 2)

            update_display(frame)

            # Track cars. Ensure we car only tracking moving vehicles and delete when out of frame
            for car in tracking_objects:
                car.age()
                car.check_health()

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
