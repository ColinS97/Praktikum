import cv2 as cv
import numpy as np
import math

img08cColor = cv.imread("CVBVP-Daten/OpenCV-08/img08c.jpg")
img08c = cv.imread("CVBVP-Daten/OpenCV-08/img08c.jpg", cv.IMREAD_GRAYSCALE)
img08d = cv.imread("CVBVP-Daten/OpenCV-08/img08d.jpg", cv.IMREAD_GRAYSCALE)

canny = cv.Canny(img08c, 100, 200, None, 3)
cv.imshow("canny", canny)
cv.waitKey(0)

lines = cv.HoughLines(canny, 1, np.pi / 180, 80, None, 0, 0)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(img08cColor, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

cv.imshow("lines", img08cColor)
cv.waitKey(0)
