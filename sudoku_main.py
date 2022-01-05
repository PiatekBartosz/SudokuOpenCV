import time as t
import numpy as np
import cv2

from helpers import pre_processing, find_corners, warp_img, isolate_cells

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

        original_img = frame.copy()

        processed_frame = pre_processing(frame)

        corners = find_corners(processed_frame, frame)

        print(corners)
        if corners:
            warp, matrix = warp_img(original_img, corners)
            processed_warp = pre_processing(warp)
            cells = isolate_cells(processed_warp)

            # todo mask the cell so that borders of each cell is deleted

        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
