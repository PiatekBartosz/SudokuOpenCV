import numpy as np
import cv2 as cv

# settings
chessboard_size = (9, 6)
cam_number = 0
print(chessboard_size[0])
print(chessboard_size[1])
cap = cv.VideoCapture(cam_number)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points shaped [[0. 0. 0.], [1. 0. 0.], [0. 0. 0.], ... , [chessboard_size[0]. chessboard_size[1]. 0.]}
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# prepare list to store img points
obj_points = []  # 3D points in real world
img_points = []  # 2D points on captured frame

while True:
    ret, img = cap.read()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chessboard corners
    ret, corners = cv.findChessboardCorners(img_gray, (chessboard_size[0], chessboard_size[1]), None)

    if ret:
        #
        obj_points.append(objp)

        corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)

        cv.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)

        # ret, mtx, rvect, tvect = cv.calibrateCamera(obj_points, img_points, img_gray.shape[::2], None, None)

    cv.imshow("OpenCV Cam Calibration", img)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
