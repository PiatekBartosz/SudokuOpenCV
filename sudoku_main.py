import time as t
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import helpers
import solver
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# select device number:
device = 0
cap = cv2.VideoCapture(device)

frameWidth = cap.get(3)
frameHeight = cap.get(4)
frame_rate = 24
cap.set(10, 150)

model = load_model('OCRmodel.h5')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

previous_time = 0

seen = dict()

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
                squares_guesses = []

                for cell in cells:
                    cell = helpers.crop_cell(cell)
                    if helpers.identify_empty(cell):
                        squares_guesses.append(0)
                    else:
                        cell = helpers.preprocess_cell(cell)
                        number, predict = helpers.validate_predict(model.predict(cell))
                        squares_guesses.append(number)
                squares_guesses = np.array_split(squares_guesses, 9)

                if solver.possible(squares_guesses):
                    try:
                        solver.solve(squares_guesses)
                    except:
                        pass
                    helpers.put_digits(warp, squares_guesses)
                    cv2.imshow('warp', warp)
                    helpers.draw_solution(warp, frame, corners, int(frameWidth), int(frameHeight))

        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
