import cv2 as cv
import numpy as np

#Aufgabe 9_01

points = []
img = cv.imread("CVBVP-Daten/OpenCV-09/Img09a.jpg")

height, width, channels = img.shape


# Cut the image in half
width_cutoff = width // 2
s1 = img[:, :width_cutoff]
s2 = img[:, width_cutoff:]


def definePoints(action, x, y, flags, param):
    if len(points) == 8 and action == cv.EVENT_LBUTTONDOWN:
        print("Points 0 bis 3: ", points[0:3])
        print("Points 4 bis 7: ", points[4:7])
        M, mask = cv.findHomography(np.float32(points[4:8]), np.float32(points[0:4]))
        # M = cv.findHomography(points[0:3], points[4:7], cv.RANSAC, 3.0)
        warpedImage = cv.warpPerspective(s1, M, (width_cutoff, height))
        cv.imshow("Warped Image", warpedImage)
    elif action == cv.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            cv.circle(s1, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("Image1: ", s1)
        else:
            cv.circle(s2, (x, y), 5, (0, 0, 255), -1)
            cv.imshow("Image2: ", s2)
        print((x, y))
        points.append((x, y))
        print(len(points))


cv.namedWindow("Image1: ")
cv.namedWindow("Image2: ")
cv.setMouseCallback("Image1: ", definePoints)
cv.setMouseCallback("Image2: ", definePoints)

k = 0

while k != 113:
    cv.imshow("Image1: ", s1)
    cv.imshow("Image2: ", s2)

    k = cv.waitKey(0)
cv.destroyAllWindows()
