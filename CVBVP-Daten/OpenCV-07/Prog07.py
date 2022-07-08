import cv2 as cv
import numpy as np

img07a = cv.imread("CVBVP-Daten/OpenCV-07/img07a.jpg", cv.IMREAD_GRAYSCALE)
img07b = cv.imread("CVBVP-Daten/OpenCV-07/img07b.jpg", cv.IMREAD_GRAYSCALE)
img07c = cv.imread("CVBVP-Daten/OpenCV-07/img07c.jpg", cv.IMREAD_GRAYSCALE)
img07d = cv.imread("CVBVP-Daten/OpenCV-07/img07d.jpg", cv.IMREAD_GRAYSCALE)

img07color = cv.imread("CVBVP-Daten/OpenCV-07/img07c.jpg")
img07color = cv.cvtColor(img07color, cv.COLOR_BGR2GRAY)


sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # 3 X 3 X-direction kernel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


ret, img07b_thresh = cv.threshold(img07b, 127, 255, cv.THRESH_BINARY)
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
# bit depth -1 means the bit depth shouldn't be changed
filtered_image_y = cv.filter2D(img07a, -1, sobel_y)
filtered_image_x = cv.filter2D(img07a, -1, sobel_x)

sobel_x1_y0 = cv.Sobel(img07a, -1, 1, 0, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
sobel_x2_y0 = cv.Sobel(img07a, -1, 2, 0, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
sobel_x1_y1 = cv.Sobel(img07a, -1, 1, 1, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
sobel_x2_y2 = cv.Sobel(img07a, -1, 2, 2, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
sobel_x0_y1 = cv.Sobel(img07a, -1, 0, 1, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)
sobel_x0_y2 = cv.Sobel(img07a, -1, 0, 2, ksize=3, scale=1, borderType=cv.BORDER_DEFAULT)

gray = np.float32(img07color)
cornerHarris = cv.cornerHarris(gray, 2, 3, 0.04)


cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_y_filtered.jpg", filtered_image_y)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_x_filtered.jpg", filtered_image_x)

cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x1_y0.jpg", sobel_x1_y0)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x2_y0.jpg", sobel_x2_y0)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x1_y1.jpg", sobel_x1_y1)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x2_y2.jpg", sobel_x2_y2)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x0_y1.jpg", sobel_x0_y1)
cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07a_sobel_x0_y2.jpg", sobel_x0_y2)

cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07c_cornerHarris.jpg", cornerHarris)


cv.imwrite("CVBVP-Daten/OpenCV-07/out/img07b_thresh.jpg", img07b_thresh)


## for 7.2 look at prakti
