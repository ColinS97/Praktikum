import cv2 as cv
import numpy as np
import time
from math import sqrt, pow

g_img = cv.imread("CVBVP-Daten/OpenCV-06/Vid06.png", cv.IMREAD_GRAYSCALE)
if not g_img.data:
    print("Cant load img")
x = 10
rows = g_img.shape[0]
cols = g_img.shape[1]
middlePoint = (int(cols / 2), int(rows / 2))
cv.imshow("raw", g_img)


def translate(img):
    tx, ty = cols / 4, rows / 4
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    translated_image = cv.warpAffine(src=img, M=translation_matrix, dsize=(cols, rows))
    cv.imshow("Image: ", translated_image)
    cv.waitKey(0)


def resize(img):
    loop = True
    while loop:
        if rows - x <= 0 or cols - x <= 0:
            x = 10
            loop = False
        print(x)
        resized = cv.resize(g_img, (rows - x, cols - x))
        cv.imshow("Image: ", resized)
        cv.waitKey(0)
        x += 10
    cv.destroyAllWindows("Image:")


def rotate(img):
    loop = True
    angle = 0
    scale = 0.5
    while angle < 360:
        rotMatrix = cv.getRotationMatrix2D(middlePoint, angle, scale)
        rotated = cv.warpAffine(g_img, rotMatrix, (cols, rows))
        cv.imshow("Image: ", rotated)
        cv.waitKey(0)
        angle += 10
    cv.destroyAllWindows("Image:")


def rotatein3d(image):
    # https://stackoverflow.com/questions/6606891/opencv-virtually-camera-rotating-translating-for-birds-eye-view/6667784#6667784
    # https://stackoverflow.com/questions/7019407/translating-and-rotating-an-image-in-3d-using-opencv
    # TODO: implement
    print("")


def warpImage(image):

    objectCornerA = [6, 250]
    objectCornerB = [309, 370]
    objectCornerC = [410, 45]
    objectCornerD = [220, 17]

    maxWidth = max(
        distance(objectCornerA, objectCornerD),
        distance(objectCornerB, objectCornerC),
    )

    maxHeight = max(
        distance(objectCornerA, objectCornerB),
        distance(objectCornerC, objectCornerD),
    )

    inputPoints = np.float32(
        [objectCornerA, objectCornerB, objectCornerC, objectCornerD]
    )

    outputPoints = np.float32(
        [
            [0, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1],
            [maxWidth - 1, 0],
        ]
    )

    m = cv.getPerspectiveTransform(inputPoints, outputPoints)

    warped = cv.warpPerspective(image, m, (maxWidth, maxHeight))

    cv.imshow("Warped", warped)

    cv.waitKey(0)


def distance(a, b):
    return int(sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2)))


translate(g_img)
