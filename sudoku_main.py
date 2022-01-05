import time as t
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import helpers

cap = cv2.VideoCapture(0)

frameWidth = cap.get(3)
frameHeight = cap.get(4)
frame_rate = 30
cap.set(10, 150)

model = load_model('OCRmodel.h5')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

previous_time = 0

while True:
    time_passage = t.time() - previous_time
    ret, frame = cap.read()

    if time_passage > 1.0 / frame_rate:
        previous_time = t.time()

        if frame is not None:
            original_img = frame.copy()

        processed_frame = helpers.pre_processing(frame)

        corners = helpers.find_corners(processed_frame, frame)

        if corners:
            warp, matrix = helpers.warp_img(original_img, corners)
            cells = helpers.isolate_cells(warp)

            if cells[0].shape[0] > 40:
                result = []

                temp = cells[75]
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                temp = cv2.equalizeHist(temp)
                cv2.imshow('', temp)

                for cell in cells:
                    cell = helpers.preprocess_cell(cell)
                    predict = model.predict(cell)

                    result.append(helpers.validate_predict(predict))

                result = np.array(result)
                result = result.reshape((9, 9))
                print(result)

        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
sudoku = np.array = [[0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 2, 0, 0, 3],
                     [0, 0, 0, 4, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 5, 0, 0],
                     [6, 0, 1, 7, 0, 0, 0, 0, 0],
                     [0, 0, 4, 1, 0, 0, 0, 0, 0],
                     [0, 5, 0, 0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 8, 0, 6, 0],
                     [0, 3, 0, 9, 1, 0, 0, 0, 0]]
