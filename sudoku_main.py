import time as t
import numpy as np
import cv2

from helpers import pre_processing, find_corners, warp_img

cap = cv2.VideoCapture(0)
frameWidth = cap.get(3)
frameHeight = cap.get(4)
frame_rate = 30
cap.set(10, 150)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

previous_time = 0

while True:
    time_passage = t.time() - previous_time

    ret, frame = cap.read()

    if time_passage > 1.0 / frame_rate:
        previous_time = t.time()

        processed_frame = pre_processing(frame)

        corners = find_corners(processed_frame, frame)

        print(corners)
        if corners:
            warp, matrix = warp_img(frame, corners)
            processed_warp = pre_processing(warp)
            cv2.imshow("Warp", processed_warp)

            # todo isolate every digit
            horizontal_stripes = np.array_split(warp, 9, axis=0)
            numbers = []
            for element in horizontal_stripes:
                stripe = np.array_split(element, 9, axis=1)
                for element2 in stripe:
                    numbers.append(element2)
            cv2.imshow("test", horizontal_stripes[0])

            for element in numbers:
                element = pre_processing(element)
            cv2.imshow("nowe", numbers[0])

        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
