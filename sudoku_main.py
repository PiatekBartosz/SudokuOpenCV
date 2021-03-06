import time as t
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import helpers
import solver
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# select device number:
device = 0
cap = cv2.VideoCapture(device)

# default camera calibration path
cameraCalibrationPath = "CalibrationData\CameraCalibrationData.json"

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30

# calibrate camera
roi, camera_matrix, distortion, new_camera_matrix = helpers.calibrate_camera(cameraCalibrationPath, frameHeight, frameWidth)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

model = load_model("CodeLabsDigitRecognition"
                   ".h5")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

previous_time = 0

seen = dict()

while True:
    time_passage = t.time() - previous_time
    ret, frame = cap.read()
    frame = helpers.undistort_camera(frame, roi, camera_matrix, distortion, new_camera_matrix)

    if time_passage > 1.0 / frame_rate:
        previous_time = t.time()

        if frame is not None:
            original_img = frame.copy()

        processed_frame = helpers.pre_processing(frame)

        corners = helpers.find_corners(processed_frame, frame)

        if corners:
            warp, matrix, inverse_matrix = helpers.warp_img(original_img, corners)
            cells = helpers.isolate_cells(warp)
            if warp.shape[0] > frameHeight * 0.5:
                squares_guesses = []

                for cell in cells:
                    cell = helpers.crop_cell(cell)
                    if helpers.identify_empty(cell):
                        squares_guesses.append(0)
                    else:
                        cell = helpers.preprocess_cell(cell)
                        model_raw_output = model.predict(cell)
                        number, probability = helpers.validate_predict(model_raw_output)
                        squares_guesses.append(number)
                squares_guesses = np.array_split(squares_guesses, 9)
                print(np.matrix(squares_guesses))

                if solver.possible(squares_guesses):
                    try:
                        solver.solve(squares_guesses)
                    except:
                        pass
                    warp_blanc = helpers.put_digits(warp, squares_guesses)
                    frame = helpers.draw_solution(warp_blanc, frame, corners, frame.shape[1], frame.shape[0], inverse_matrix)
        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
