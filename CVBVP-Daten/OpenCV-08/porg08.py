import cv2 as cv
import numpy as np

img08a = cv.imread("CVBVP-Daten/OpenCV-08/img08a.jpg")
img08b = cv.imread("CVBVP-Daten/OpenCV-08/img08b.jpg", cv.IMREAD_GRAYSCALE)


dilatation_size = 3
erosion_size = 2

cv.imshow("base", img08a)
cv.waitKey(0)
gray = cv.cvtColor(img08a, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("thresh", thresh)
cv.waitKey(0)

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
cv.imshow("opening", opening)
cv.waitKey(0)
# sure background area
element = cv.getStructuringElement(
    cv.MORPH_ELLIPSE,
    (2 * dilatation_size + 1, 2 * dilatation_size + 1),
    (dilatation_size, dilatation_size),
)
sure_bg = cv.dilate(thresh, element)
cv.imshow("sure_bg", sure_bg)
cv.waitKey(0)
# Finding sure foreground area
element = cv.getStructuringElement(
    cv.MORPH_ELLIPSE,
    (2 * erosion_size + 1, 2 * erosion_size + 1),
    (erosion_size, erosion_size),
)

sure_fg = cv.erode(opening, element)
cv.imshow("sure_fg", sure_fg)
cv.waitKey(0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)


# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

img08awater = img08a.copy()
markers = cv.watershed(img08awater, markers)
img08awater[markers == -1] = [255, 0, 0]
cv.imshow("watershed", img08awater)
cv.waitKey(0)

### now finding contours
blurred = cv.blur(gray, (7, 7))
edges = cv.Canny(gray, 100, 200)
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

outercontours = []
for contour in contours:
    area = cv.contourArea(contour)
    # hier wwaere ein check sinnvoller ob eine area in der andern drin liet aber ich bin fauk und das hier funktioniert fast gut genug
    if area > 100:
        outercontours.append(contour)
img08a_contours = cv.drawContours(img08a, outercontours, -1, (0, 255, 0), 3)
cv.imshow("contours", img08a_contours)
cv.waitKey(0)

# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
def centroid(contour):
    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


# draw bounding rect around contours
img08a_contours_rect = img08a.copy()
rects = []
for contour in outercontours:
    x, y = centroid(contour)
    img08a_contours_rect = cv.circle(
        img08a_contours_rect, (int(x), int(y)), 5, (0, 0, 255), -1
    )
    x, y, w, h = cv.boundingRect(contour)
    img08a_contours_rect = cv.rectangle(
        img08a_contours_rect, (x, y), (x + w, y + h), (0, 255, 0), 2
    )
cv.imshow("contours_rect", img08a_contours_rect)
cv.waitKey(0)
