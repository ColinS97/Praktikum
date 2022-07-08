import cv2 as cv
import numpy as np

## 1
img04a01 = cv.imread("CVBVP-Daten/OpenCV-04/Img04a01.jpg")
img04a02 = cv.imread("CVBVP-Daten/OpenCV-04/Img04a02.jpg")


stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)

status, img04aStitched = stitcher.stitch([img04a01, img04a02])
print(status)
print(cv.Stitcher_OK)
cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04a_stitched.jpg", img04aStitched)
cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04a_blurred.jpg", cv.blur(img04a01, (5, 5)))
cv.imwrite(
    "CVBVP-Daten/OpenCV-04/out/Img04a_medianblur.jpg", cv.medianBlur(img04a01, (5))
)
cv.imshow("Img04a_stitched", img04aStitched)

## 2


img04d01 = cv.imread("CVBVP-Daten/OpenCV-04/Img04d01.jpeg")
img04d02 = cv.imread("CVBVP-Daten/OpenCV-04/Img04d02.jpeg")

brisk = cv.BRISK_create()

kp1, des1 = brisk.detectAndCompute(img04d01, None)
kp2, des2 = brisk.detectAndCompute(img04d02, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(
    img04d01,
    kp1,
    img04d02,
    kp2,
    matches[:10],
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv.drawKeypoints(img04d01, kp1, img04d01, color=(255, 0, 0))
cv.drawKeypoints(img04d02, kp2, img04d02, color=(255, 0, 0))

cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04d01_kp.jpg", img04d01)
cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04d02_kp.jpg", img04d02)

cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04d_matches.jpg", img3)

## 3

img04b = cv.imread("CVBVP-Daten/OpenCV-04/Img04b.jpg", cv.IMREAD_GRAYSCALE)
img04c = cv.imread("CVBVP-Daten/OpenCV-04/Img04c.png", cv.IMREAD_GRAYSCALE)


face_cascade_name = "CVBVP-Daten/OpenCV-04/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "CVBVP-Daten/OpenCV-04/haarcascade_eye.xml"
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
face_cascade.load(face_cascade_name)
eyes_cascade.load(eyes_cascade_name)

faces = face_cascade.detectMultiScale(img04b)
for (x, y, w, h) in faces:
    center = (x + w // 2, y + h // 2)
    img04b = cv.ellipse(img04b, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = img04b[y : y + h, x : x + w]
    # -- In each face, detect eyes
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2, y2, w2, h2) in eyes:
        eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        radius = int(round((w2 + h2) * 0.25))
        img04b = cv.circle(img04b, eye_center, radius, (255, 0, 0), 4)
cv.imwrite("CVBVP-Daten/OpenCV-04/out/Img04b_faces.jpg", img04b)
