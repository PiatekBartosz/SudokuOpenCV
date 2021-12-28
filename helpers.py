import cv2
import numpy as np


def preProcessing(img):
    # https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    imgInvert = cv2.bitwise_not(imgThreshold, 0)
    imgKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    imgMorph = cv2.morphologyEx(imgInvert, cv2.MORPH_OPEN, imgKernel)
    result = cv2.dilate(imgMorph, imgKernel, iterations=1)
    return result

def findContours(processed_frame):
    # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    # using '_' to ignore 'hierarchy' output we will not use
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # we need to find the largest contour that has 4 corners
    # https://learnopencv.com/contour-detection-using-opencv-python-c/
    for i in contours:
        area = cv2.contourArea(i)
        perimeter = cv2.arcLength(i, closed=True)
        approx = cv2.approxPolyDP(i, 0.01 * perimeter, closed=True)

        if len(approx) == 4 and area > 1000:
            biggest = approx
            break


    print(biggest)
    return biggest
