import time as t
import numpy as np
import cv2

from helpers import preProcessing, findContours

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
 # uncomment after finishing findContours()
    #if time_passage > 1.0 / frame_rate:
    previous_time = t.time()

    processed_frame = preProcessing(frame)

    largest_item = findContours(processed_frame)

    result_frame = processed_frame
    # delete after finishing findContours()
    print(largest_item)
    cv2.drawContours(frame, largest_item, -1, (255, 0, 0), 10)

    cv2.imshow("SudokuOpenCV", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
