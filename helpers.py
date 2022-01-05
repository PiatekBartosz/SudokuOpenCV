import cv2
import numpy as np


def pre_processing(img):
    # https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    imgInvert = cv2.bitwise_not(imgThreshold, 0)
    imgKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    imgMorph = cv2.morphologyEx(imgInvert, cv2.MORPH_OPEN, imgKernel)
    result = cv2.dilate(imgMorph, imgKernel, iterations=1)
    return result


def find_corners(processed_frame, frame):
    # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    # using '_' to ignore 'hierarchy' output we will not use
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # we need to find the largest contour that has 4 corners
    # https://learnopencv.com/contour-detection-using-opencv-python-c/
    for i in sorted_contours:
        area = cv2.contourArea(i)
        perimeter = cv2.arcLength(i, closed=True)
        approx = cv2.approxPolyDP(i, 0.01 * perimeter, closed=True)

        # check if it has 4 corners
        if len(approx) == 4 and area > 1000:
            biggest = approx
            break

    if biggest is not None:
        # get corners
        top_left = find_corner_position(biggest, min, np.add)
        top_right = find_corner_position(biggest, max, np.subtract)
        bot_left = find_corner_position(biggest, min, np.subtract)
        bot_right = find_corner_position(biggest, max, np.add)

        # check for square if b!= 0 or a/b ~ 1
        if bot_right[1] - top_right[1] == 0:
            return None
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return None

        # draw found square
        cv2.drawContours(frame, biggest, -1, (255, 0, 0), 10)
        return [top_left, top_right, bot_right, bot_left]

    return None


def find_corner_position(corners, limit, compare):
    # limit: min, max
    # compare: np.add, np.subtract
    result, _ = limit(enumerate([compare(cr[0][0], cr[0][1]) for cr in corners]),
                      key=lambda x: x[1])
    return corners[result][0][0], corners[result][0][1]


def warp_img(frame, corners):
    # https://theailearner.com/tag/cv2-getperspectivetransform/

    corners = np.array(corners, dtype='float32')

    top_left, top_right, bot_right, bot_left = corners

    width = int(max([
        np.abs((top_right[0] - bot_right[0]) ** 2 + (top_right[1] - bot_right[1]) ** 2) ** 0.5,
        np.abs((top_left[0] - bot_left[0]) ** 2 + (top_left[1] - bot_left[1]) ** 2) ** 0.5,
        np.abs((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2) ** 0.5,
        np.abs((bot_left[0] - bot_right[0]) ** 2 + (bot_left[1] - bot_right[1]) ** 2) ** 0.5
    ]))

    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(frame, matrix, (width, width)), matrix


def isolate_cells(warp):
    horizontal_stripes = np.array_split(warp, 9, axis=0)
    cells = []
    for stripe in horizontal_stripes:
        cells.extend(np.array_split(stripe, 9, axis=1))
    return cells


def preprocess_cell(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, 0)
    return img


def validate_predict(cell):
    cell = cell.tolist()[0]
    max_value = max(cell)

    if max_value > 0.8:
        return cell.index(max_value) + 1
    else:
        return 0
