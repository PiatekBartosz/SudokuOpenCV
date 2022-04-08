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
            cv2.imshow("warp", warp)
            if warp.shape[0] > frameHeight * 0.5:
                squares_guesses = []

                for cell in cells:
                    cell = helpers.crop_cell(cell)
                    if helpers.identify_empty(cell):
                        squares_guesses.append('b')
                    else:
                        cv2.imshow("cropped", cell)
                        cell = helpers.preprocess_cell(cell)
                        model_raw_output = model.predict(cell)
                        number = np.argmax(model_raw_output, axis=1)
                        squares_guesses.append(number[0])
                squares_guesses = np.array_split(squares_guesses, 9)
                print(squares_guesses)

                # if solver.possible(squares_guesses):
                #     try:
                #         solver.solve(squares_guesses)
                #     except:
                #         pass
                    # helpers.put_digits(warp, squares_guesses)
                    # solution = helpers.draw_solution(warp, frame, corners, int(frameWidth), int(frameHeight), inverse_matrix)
                    # cv2.imshow("soulution", solution)

        cv2.imshow("SudokuOpenCV", frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
