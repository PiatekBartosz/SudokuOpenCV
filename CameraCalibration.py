import numpy as np
import cv2 as cv
import time as t

# settings
chessboard_size = (6, 5)
cam_number = 0

cap = cv.VideoCapture(cam_number)
frame_rate = 30
previous_time = 0
width = cap.get(3)
height = cap.get(4)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points shaped [[0. 0. 0.], [1. 0. 0.], [0. 0. 0.], ... , [chessboard_size[0]. chessboard_size[1]. 0.]}
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# prepare list to store img points
obj_points = []  # 3D points in real world
img_points = []  # 2D points on captured frame

while True:
    # camera frame rate
    time_elapsed = t.time() - previous_time
    ret, img = cap.read()

    if time_elapsed > 1.0 / frame_rate:
        previous_time = t.time()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv.findChessboardCorners(img_gray, (chessboard_size[0], chessboard_size[1]), None)

        if ret:
            obj_points.append(objp)

            corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            cv.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)

            # camera calibrated bool, camera matrix, distortion parameters, rotation vectors, translation vectors
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, img.shape[:2], None, None)

            newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

            # undistort
            dst = cv.undistort(img, mtx, dist, None, newCameraMatrix)

            # crop the img
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]


    cv.imshow("OpenCV Cam Calibration", img)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
